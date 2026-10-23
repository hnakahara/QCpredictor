"""
QC予測の推論ロジック（Webアプリ用）

predict_qc.py と同じ前処理・同じモデルで予測し、
判定ラベルに加えて Qualified 確率を返す。
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"

# モデルに渡す入力列
INPUT_COLUMNS = ["TN_HU", "CancerType", "OpeBx", "StorageTime"]
CATEGORICAL_COLUMNS = ["CancerType", "OpeBx"]
POSITIVE_LABEL = "Qualified"  # 確率を表示する側のラベル
MODEL_ORDER = ["GBM", "RF", "NN", "SVM"]  # 画面・CSVでの表示順


# 学習データの保存期間の範囲（日）。これを超える入力には警告を出す
STORAGE_TRAIN_MIN = 0
STORAGE_TRAIN_MAX = 4714


def storage_days(collection_date: date, submission_date: date) -> int:
    """保存期間（日数）= 提出日 − 取得日。同日なら0日。"""
    return (submission_date - collection_date).days


class NumpyMLP:
    """
    Kerasの全結合NNを numpy だけで計算する（TensorFlow不要）。
    重みは tools/export_nn_weights.py で nn_qc_model.keras から書き出したもの。
    Dropout は推論時には何もしないため省略している。
    """

    def __init__(self, path: Path):
        with np.load(path) as z:
            self.activations = [str(a) for a in z["activations"]]
            self.weights = [(z[f"W{i}"], z[f"b{i}"]) for i in range(len(self.activations))]

    def predict_proba(self, X) -> np.ndarray:
        h = np.asarray(X, dtype=np.float32)
        for (W, b), act in zip(self.weights, self.activations):
            h = h @ W + b
            if act == "relu":
                h = np.maximum(h, 0)
            elif act == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-h))
            elif act != "linear":
                raise ValueError(f"Unsupported activation: {act}")
        return h.ravel()


class QCPredictor:
    """4モデル（GBM / RF / SVM / NN）をまとめて読み込み、予測する。"""

    def __init__(self, model_dir: Path = MODEL_DIR):
        # GBM
        gbm = joblib.load(model_dir / "gbm_qc_pipeline.joblib")
        self.gbm = gbm["model"]
        self.gbm_label_encoders = gbm["label_encoders"]
        self.gbm_y_encoder = gbm["y_encoder"]
        self.gbm_features = gbm["features"]

        # RF
        rf = joblib.load(model_dir / "rf_qc_pipeline.joblib")
        self.rf = rf["model"]
        self.rf_label_encoders = rf["label_encoders"]
        self.rf_y_encoder = rf["y_encoder"]
        self.rf_features = rf["features"]

        # SVM
        svm = joblib.load(model_dir / "svm_qc_pipeline.joblib")
        self.svm = svm["model"]
        self.svm_scaler = svm["scaler"]
        self.svm_feature_columns = svm["feature_columns"]

        # NN（numpyで計算する。TensorFlowは使わない）
        self.nn = NumpyMLP(model_dir / "nn_qc_weights.npz")
        nn_pre = joblib.load(model_dir / "nn_qc_preprocess.joblib")
        self.nn_scaler = nn_pre["scaler"]
        self.nn_label_encoder = nn_pre["label_encoder_qc"]
        self.nn_feature_columns = nn_pre["feature_columns"]

        # 選択肢は学習時のエンコーダーから取得する（学習時にないカテゴリは入力させない）
        self.cancer_types = list(self.gbm_label_encoders["CancerType"].classes_)
        self.ope_bx = list(self.gbm_label_encoders["OpeBx"].classes_)

    # ------------------------------------------------------------
    # 前処理（predict_qc.py と同一）
    # ------------------------------------------------------------
    @staticmethod
    def _label_encode(df, encoders, features):
        df = df.copy()
        for col in CATEGORICAL_COLUMNS:
            df[col] = encoders[col].transform(df[col])
        return df[features]

    @staticmethod
    def _one_hot_scale(df, feature_columns, scaler):
        X = pd.get_dummies(df[INPUT_COLUMNS], columns=CATEGORICAL_COLUMNS, drop_first=True)
        X = X.reindex(columns=feature_columns, fill_value=0)
        return scaler.transform(X)

    @staticmethod
    def _positive_proba(model, X, y_encoder) -> np.ndarray:
        """GBM / RF の Qualified 確率（数値ラベル→列番号を対応付けて取り出す）"""
        positive_code = y_encoder.transform([POSITIVE_LABEL])[0]
        col = list(model.classes_).index(positive_code)
        return model.predict_proba(X)[:, col]

    # ------------------------------------------------------------
    # 予測
    # ------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        各モデルの判定（*_pred）と Qualified 確率（*_prob）を返す。
        SVMは判定と確率が一致しないため（class_weight='balanced' と
        Plattスケーリングの差）、確率は返さない。
        """
        missing = set(INPUT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        out = {}

        # GBM
        X = self._label_encode(df, self.gbm_label_encoders, self.gbm_features)
        out["GBM_pred"] = self.gbm_y_encoder.inverse_transform(self.gbm.predict(X))
        out["GBM_prob"] = self._positive_proba(self.gbm, X, self.gbm_y_encoder)

        # RF
        X = self._label_encode(df, self.rf_label_encoders, self.rf_features)
        out["RF_pred"] = self.rf_y_encoder.inverse_transform(self.rf.predict(X))
        out["RF_prob"] = self._positive_proba(self.rf, X, self.rf_y_encoder)

        # SVM（判定のみ）
        X = self._one_hot_scale(df, self.svm_feature_columns, self.svm_scaler)
        out["SVM_pred"] = self.svm.predict(X)
        out["SVM_prob"] = np.full(len(df), np.nan)

        # NN（シグモイド出力 = 2番目のクラス = Qualified の確率、閾値0.5）
        X = self._one_hot_scale(df, self.nn_feature_columns, self.nn_scaler)
        prob = self.nn.predict_proba(X)
        assert self.nn_label_encoder.classes_[1] == POSITIVE_LABEL
        out["NN_pred"] = self.nn_label_encoder.inverse_transform((prob > 0.5).astype(int))
        out["NN_prob"] = prob

        return pd.DataFrame(out, index=df.index)

    def predict_one(self, tn_hu: float, cancer_type: str, ope_bx: str, storage_time: int) -> pd.DataFrame:
        """1件分を予測し、モデルごとの行（model / pred / prob）で返す。"""
        row = pd.DataFrame(
            [{"TN_HU": tn_hu, "CancerType": cancer_type, "OpeBx": ope_bx, "StorageTime": storage_time}]
        )
        res = self.predict(row).iloc[0]
        return pd.DataFrame(
            [
                {"model": m, "pred": res[f"{m}_pred"], "prob": res[f"{m}_prob"]}
                for m in MODEL_ORDER
            ]
        )


# ============================================================
# CSV一括入力
# ============================================================
MAX_BATCH_ROWS = 1000  # 1回に受け付ける最大件数

# CSVの列名の別名（大文字小文字・空白・アンダースコアは無視して照合する）
COLUMN_ALIASES = {
    "TN_HU": ["tnhu", "tumornuclei", "腫瘍細胞含有率"],
    "CancerType": ["cancertype", "がん種"],
    "OpeBx": ["opebx", "sampletype", "検体の種類"],
    "CollectionDate": ["collectiondate", "検体取得日", "取得日"],
    "SubmissionDate": ["submissiondate", "検体提出日", "提出日"],
    "StorageTime": ["storagetime", "保存期間"],
}
DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"]

# 行ごとの判定結果コード（画面側で日英の文言に変換する）
ERROR_CODES = [
    "missing_tn", "tn_not_number", "tn_range",
    "missing_cancer", "unknown_cancer",
    "missing_opebx", "unknown_opebx",
    "missing_storage", "bad_date", "date_order", "storage_invalid",
]
WARNING_CODES = ["storage_range"]


def _norm_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("（%）", "").replace("(%)", "")


def normalize_columns(df: pd.DataFrame) -> dict:
    """CSVの列名 → 正式な列名 の対応表を返す（見つかった列のみ）"""
    lookup = {alias: canon for canon, aliases in COLUMN_ALIASES.items() for alias in aliases}
    mapping = {}
    for col in df.columns:
        canon = lookup.get(_norm_key(col))
        if canon and canon not in mapping.values():
            mapping[col] = canon
    return mapping


def _parse_date(value):
    """日付文字列を date に変換する。空欄は None、読めなければ NaT 相当として False を返す"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split(" ")[0].split("T")[0]  # 時刻付き（2024-01-01 00:00:00）は日付部分だけ使う
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return False


def _blank(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


class BatchError(ValueError):
    """CSV全体として受け付けられない場合（列不足・件数超過・空ファイル）"""

    def __init__(self, code: str, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """UTF-8（BOM付き含む）→ Shift-JIS の順で読み込む。値はすべて文字列として読む"""
    import io

    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise BatchError("encoding")


def validate_batch(df: pd.DataFrame, cancer_types: list, ope_bx: list) -> pd.DataFrame:
    """
    行ごとに入力を検証し、モデル入力（TN_HU / CancerType / OpeBx / StorageTime）と
    判定結果コード（codes：エラー・警告のリスト）を返す。

    保存期間は CollectionDate と SubmissionDate があればそこから計算し、
    なければ StorageTime 列（日数）を使う。
    """
    # 全列が空の行は無視する
    df = df[~df.apply(lambda r: all(_blank(v) for v in r), axis=1)]
    if len(df) == 0:
        raise BatchError("empty")
    if len(df) > MAX_BATCH_ROWS:
        raise BatchError("too_many", len(df))

    mapping = normalize_columns(df)
    have = set(mapping.values())
    missing = [c for c in ["TN_HU", "CancerType", "OpeBx"] if c not in have]
    has_dates = {"CollectionDate", "SubmissionDate"} <= have
    if not has_dates and "StorageTime" not in have:
        missing.append("CollectionDate / SubmissionDate")
    if missing:
        raise BatchError("missing_columns", missing)

    src = df.rename(columns=mapping)
    cancer_lookup = {c.lower(): c for c in cancer_types}
    opebx_lookup = {c.lower(): c for c in ope_bx}

    rows = []
    for _, r in src.iterrows():
        codes = []
        out = {"TN_HU": np.nan, "CancerType": None, "OpeBx": None, "StorageTime": None}

        # TN_HU（0〜100）
        if _blank(r["TN_HU"]):
            codes.append("missing_tn")
        else:
            tn = pd.to_numeric(str(r["TN_HU"]).strip().rstrip("%"), errors="coerce")
            if pd.isna(tn):
                codes.append("tn_not_number")
            elif not 0 <= tn <= 100:
                codes.append("tn_range")
            else:
                out["TN_HU"] = float(tn)

        # がん種（大文字小文字・空白の違いは許容）
        if _blank(r["CancerType"]):
            codes.append("missing_cancer")
        else:
            key = str(r["CancerType"]).strip().replace(" ", "_").lower()
            if key in cancer_lookup:
                out["CancerType"] = cancer_lookup[key]
            else:
                codes.append("unknown_cancer")

        # 手術 / 生検
        if _blank(r["OpeBx"]):
            codes.append("missing_opebx")
        else:
            key = str(r["OpeBx"]).strip().lower()
            if key in opebx_lookup:
                out["OpeBx"] = opebx_lookup[key]
            else:
                codes.append("unknown_opebx")

        # 保存期間：日付があれば日付から、なければ StorageTime から
        d1 = _parse_date(r["CollectionDate"]) if has_dates else None
        d2 = _parse_date(r["SubmissionDate"]) if has_dates else None
        if d1 is False or d2 is False:
            codes.append("bad_date")
        elif d1 is not None and d2 is not None:
            days = storage_days(d1, d2)
            if days < 0:
                codes.append("date_order")
            else:
                out["StorageTime"] = days
        elif "StorageTime" in src.columns and not _blank(r["StorageTime"]):
            st = pd.to_numeric(str(r["StorageTime"]).strip(), errors="coerce")
            if pd.isna(st) or st < 0:
                codes.append("storage_invalid")
            else:
                out["StorageTime"] = int(round(st))  # 日数は整数にそろえる
        else:
            codes.append("missing_storage")

        if out["StorageTime"] is not None and out["StorageTime"] > STORAGE_TRAIN_MAX:
            codes.append("storage_range")

        out["codes"] = codes
        rows.append(out)

    return pd.DataFrame(rows, index=df.index)


def predict_batch(predictor: "QCPredictor", raw: pd.DataFrame) -> pd.DataFrame:
    """
    CSVの元データに、保存期間・各モデルの判定と確率・判定結果コードを付けて返す。
    エラーのある行は予測せず空欄にする。
    """
    checked = validate_batch(raw, predictor.cancer_types, predictor.ope_bx)
    raw = raw.loc[checked.index]
    ok = checked["codes"].apply(lambda c: not any(code in ERROR_CODES for code in c))

    result = raw.copy()
    result["StorageTime"] = checked["StorageTime"].astype("Int64")
    for m in MODEL_ORDER:
        result[f"{m}_pred"] = None
        if m != "SVM":
            result[f"{m}_prob_Qualified"] = np.nan

    if ok.any():
        X = checked.loc[ok, INPUT_COLUMNS].astype({"TN_HU": float, "StorageTime": float})
        pred = predictor.predict(X)
        for m in MODEL_ORDER:
            result.loc[ok, f"{m}_pred"] = pred[f"{m}_pred"].values
            if m != "SVM":
                result.loc[ok, f"{m}_prob_Qualified"] = pred[f"{m}_prob"].astype(float).round(4).values

    result["codes"] = checked["codes"]
    return result.reset_index(drop=True)
