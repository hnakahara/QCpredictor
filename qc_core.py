"""
QC予測の推論ロジック（Webアプリ用）

predict_qc.py と同じ前処理・同じモデルで予測し、
判定ラベルに加えて Qualified 確率を返す。
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"

# モデルに渡す入力列
INPUT_COLUMNS = ["TN_HU", "CancerType", "OpeBx", "StorageTime"]
CATEGORICAL_COLUMNS = ["CancerType", "OpeBx"]
POSITIVE_LABEL = "Qualified"  # 確率を表示する側のラベル


def storage_days(collection_date: date, submission_date: date) -> int:
    """保存期間（日数）= 提出日 − 取得日。同日なら0日。"""
    return (submission_date - collection_date).days


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

        # NN（TensorFlowは重いので、ここで初めてimportする）
        from tensorflow.keras.models import load_model

        self.nn = load_model(model_dir / "nn_qc_model.keras")
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
        prob = self.nn.predict(X, verbose=0).ravel()
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
                for m in ["GBM", "RF", "SVM", "NN"]
            ]
        )
