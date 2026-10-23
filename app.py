"""
QC予測 Webアプリ（Streamlit）

入力5項目（TN_HU・がん種・手術/生検・検体取得日・検体提出日）から
保存期間を算出し、4モデル（GBM / RF / NN / SVM）のQC判定を表示する。
CSVファイルによる一括入力（最大1,000件）にも対応する。
"""
import time
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

import notify
from qc_core import (
    ERROR_CODES,
    MAX_BATCH_ROWS,
    MODEL_ORDER,
    STORAGE_TRAIN_MAX,
    BatchError,
    QCPredictor,
    predict_batch,
    read_csv_bytes,
    storage_days,
)

REPO_URL = "https://github.com/hnakahara/QCpredictor"
REFERENCE = "Nakahara H, et al. NPJ Genom Med. 2026"
REFERENCE_URL = "https://doi.org/10.1038/s41525-026-00595-8"
# 画面に表示する引用表記（コピーボタン付き）
CITATION = (
    "Nakahara H, Niitsu H, Motonaga M, Matsuo H, Arihiro K, Hayes CN, Hinoi T. "
    "Specimen quality shapes the actionable genomic landscape in comprehensive cancer genomic profiling. "
    "NPJ Genom Med. 2026 Jul 11. doi: 10.1038/s41525-026-00595-8. Epub ahead of print. PMID: 42436136."
)
PREVIEW_ROWS = 5  # 一括入力で画面に表示する件数
CONTACT_MAX_CHARS = 2000  # お問い合わせ内容の最大文字数
CONTACT_INTERVAL_SEC = 60  # 同じ利用者が続けて送るときの間隔
CONTACT_PER_SESSION = 3  # 1回の利用（ブラウザのセッション）で送れる件数

TEMPLATE_CSV = (
    "TN_HU,CancerType,OpeBx,CollectionDate,SubmissionDate,StorageTime\n"
    "15,Adrenal_Gland,Bx,2024/01/01,2025/04/05,\n"  # 日付で入力する例
    "60,Lung,Ope,,,209\n"  # StorageTime（日数）で入力する例
)

# ------------------------------------------------------------
# 表示文言（日本語 / 英語）
# ------------------------------------------------------------
TEXT = {
    "ja": {
        "page_title": "QC予測ツール",
        "title": "QC予測ツール",
        "lead": "検体の情報を入力すると、4つの機械学習モデルがQCチェックの見込みを判定します。",
        "reference": "組織検体を用いたがんゲノムプロファイリング検査の検査品質評価のために作成しました（{ref}）。",
        "cite": "本ツールを使用して研究を行う場合は、以下の論文の引用をお願いします。",
        "mode_single": "1件ずつ入力",
        "mode_batch": "CSVで一括入力",
        "tn_hu": "腫瘍細胞含有率（%）",
        "tn_hu_help": "病理医が評価した腫瘍細胞含有率（0〜100）",
        "cancer_type": "がん種",
        "ope_bx": "検体の種類",
        "collection": "検体取得日",
        "submission": "検体提出日",
        "storage_help": "保存期間は「提出日 − 取得日」の日数で自動計算します。",
        "select": "選択してください",
        "submit": "予測する",
        "loading": "モデルを読み込んでいます…",
        "err_missing": "未入力の項目があります：",
        "err_order": "検体提出日は検体取得日と同じ日か、それ以降にしてください。",
        "warn_range": "保存期間が学習データの範囲（0〜{max:,}日）を超えています。予測の信頼性が低い可能性があります。",
        "result": "予測結果",
        "storage_line": "**保存期間：{days:,}日**",
        "model": "モデル",
        "summary": "入力内容：",
        "verdict": "判定",
        "prob": "Qualified確率",
        "prob_na": "―",
        "mark": "※",
        "svm_note": "※ SVMは判定と確率の計算方法が異なり、両者が一致しないため確率を表示していません。",
        "legend_pass": "QCチェックをPassする可能性が高い",
        "legend_qualified": "検出感度が下がる可能性あり",
        "disclaimer": "本研究ツールは研究用途に作成しました。表示される判定は、実際の検査品質を保証するものではありません。パネル選択には臨床情報をふまえて、総合的にご判断ください。なお入力内容はサーバーには保存されません。ツールが正常に動作しているかを確認するため、解析の実行時には管理者にメールで通知しています（1件ずつの入力では入力内容と判定結果を含みます）。",
        "license": "モデルとコードは CC BY-NC（非営利）ライセンスで公開されています。",
        "source": "ソースコード",
        "lang": "Language",
        "models": {
            "GBM": "勾配ブースティング",
            "RF": "ランダムフォレスト",
            "SVM": "サポートベクターマシン",
            "NN": "ニューラルネットワーク",
        },
        "ope_bx_labels": {"Ope": "手術検体（Ope）", "Bx": "生検検体（Bx）"},
        # お問い合わせ
        "contact": "お問い合わせ",
        "contact_intro": "ツールについてのご質問・ご意見は、こちらからお送りください。いただいた情報は、お問い合わせへの回答のためにのみ使用します。",
        "contact_name": "お名前（必須）",
        "contact_affiliation": "ご所属（必須）",
        "contact_email": "メールアドレス（必須・返信用）",
        "contact_message": "お問い合わせ内容（必須・2,000字まで）",
        "contact_submit": "送信する",
        "contact_sent": "お問い合わせを送信しました。ありがとうございました。",
        "contact_errors": {
            "name": "お名前を入力してください。",
            "affiliation": "ご所属を入力してください。",
            "email": "メールアドレスを正しく入力してください。",
            "message": "お問い合わせ内容を入力してください。",
            "too_long": "お問い合わせ内容は2,000字以内にしてください。",
            "interval": "続けて送信する場合は、1分ほど時間をおいてください。",
            "session_limit": "一度に送信できるのは3件までです。時間をおいて再度お試しください。",
            "limit": "本日の受付件数の上限に達しました。明日以降に再度お試しください。",
            "unavailable": "現在お問い合わせを受け付けていません。",
            "failed": "送信に失敗しました。時間をおいて再度お試しください。",
        },
        # CSV一括入力
        "batch_intro": "CSVファイルで最大{max:,}件をまとめて予測できます。結果は画面に先頭{n}件を表示し、全件はCSVでダウンロードできます。",
        "batch_columns": (
            "| 列名 | 内容 |\n| --- | --- |\n"
            "| TN_HU | 腫瘍細胞含有率（0〜100） |\n"
            "| CancerType | がん種（下の一覧の英語名） |\n"
            "| OpeBx | Ope（手術検体）または Bx（生検検体） |\n"
            "| CollectionDate | 検体取得日（例：2024/01/01） |\n"
            "| SubmissionDate | 検体提出日（例：2025/04/05） |\n"
            "| StorageTime | 保存期間（日数）。取得日・提出日の代わりに使う場合のみ |\n\n"
            "保存期間は、取得日・提出日か StorageTime のどちらか一方を入力します。両方ある行は日付から計算します。"
            "文字コードは UTF-8 と Shift-JIS に対応しています。"
        ),
        "format": "CSVの形式（列名とがん種の一覧）",
        "cancer_list": "がん種の一覧（CancerType に入力する名前）",
        "template": "入力用テンプレート（CSV）をダウンロード",
        "upload": "CSVファイルを選択",
        "batch_summary": "{total:,}件中 {ok:,}件を予測しました（エラー {err:,}件、警告 {warn:,}件）。",
        "batch_has_errors": "エラーのある行は予測していません。理由は結果CSVの Message 列に記載しています。",
        "preview": "先頭{n}件の結果",
        "row": "行",
        "message": "メッセージ",
        "download": "結果をCSVでダウンロード（全{total:,}件）",
        "file_errors": {
            "empty": "CSVにデータがありません。",
            "too_many": "件数が上限を超えています（{detail:,}件）。1回に{max:,}件までにしてください。",
            "missing_columns": "必要な列がありません：{detail}",
            "encoding": "CSVの文字コードを読み取れません。UTF-8 または Shift-JIS で保存してください。",
            "parse": "CSVとして読み込めませんでした。ファイルの形式を確認してください。",
        },
        "row_messages": {
            "missing_tn": "TN_HU が空欄です",
            "tn_not_number": "TN_HU が数値ではありません",
            "tn_range": "TN_HU が0〜100の範囲外です",
            "missing_cancer": "CancerType が空欄です",
            "unknown_cancer": "CancerType が一覧にないがん種です",
            "missing_opebx": "OpeBx が空欄です",
            "unknown_opebx": "OpeBx は Ope か Bx で入力してください",
            "missing_storage": "取得日・提出日（または StorageTime）が空欄です",
            "bad_date": "日付を読み取れません（例：2024/01/01）",
            "date_order": "提出日が取得日より前です",
            "storage_invalid": "StorageTime が0以上の数値ではありません",
            "storage_range": "警告：保存期間が学習データの範囲（0〜{max:,}日）を超えています",
        },
    },
    "en": {
        "page_title": "QC Predictor",
        "title": "QC Predictor",
        "lead": "Enter the sample information and four machine learning models will predict its QC outcome.",
        "reference": "Developed to evaluate the quality of comprehensive genomic profiling tests using tissue specimens ({ref}).",
        "cite": "If you use this tool in your research, please cite the following paper:",
        "mode_single": "Single sample",
        "mode_batch": "Batch (CSV)",
        "tn_hu": "Tumor nuclei (%)",
        "tn_hu_help": "Percent tumor nuclei assessed by a pathologist (0–100)",
        "cancer_type": "Cancer type",
        "ope_bx": "Sample type",
        "collection": "Collection date",
        "submission": "Submission date",
        "storage_help": "Storage time is calculated automatically as submission date − collection date, in days.",
        "select": "Select",
        "submit": "Predict",
        "loading": "Loading models…",
        "err_missing": "Please fill in: ",
        "err_order": "The submission date must be on or after the collection date.",
        "warn_range": "The storage time is outside the range of the training data (0–{max:,} days). The predictions may be less reliable.",
        "result": "Results",
        "storage_line": "**Storage time: {days:,} days**",
        "model": "Model",
        "summary": "Input: ",
        "verdict": "Prediction",
        "prob": "Probability of Qualified",
        "prob_na": "—",
        "mark": "(see note)",
        "svm_note": "Note: SVM computes its prediction and its probability differently, so the two can disagree. Its probability is therefore not shown.",
        "legend_pass": "likely to pass the QC check",
        "legend_qualified": "detection sensitivity may be reduced",
        "disclaimer": "This tool was developed for research purposes. The predictions shown do not guarantee the actual quality of the test. Please decide on panel selection comprehensively, taking clinical information into account. Inputs are not stored on the server. To confirm that the tool is working correctly, the administrator is notified by email when an analysis is run (for single-sample input, the email includes the inputs and predictions).",
        "license": "The models and code are released under the CC BY-NC (non-commercial) license.",
        "source": "Source code",
        "lang": "言語",
        "models": {
            "GBM": "Gradient Boosting",
            "RF": "Random Forest",
            "SVM": "Support Vector Machine",
            "NN": "Neural Network",
        },
        "ope_bx_labels": {"Ope": "Surgical (Ope)", "Bx": "Biopsy (Bx)"},
        # お問い合わせ
        "contact": "Contact",
        "contact_intro": "Send questions or comments about this tool here. The information you provide is used only to respond to your inquiry.",
        "contact_name": "Name (required)",
        "contact_affiliation": "Affiliation (required)",
        "contact_email": "Email address (required, for our reply)",
        "contact_message": "Message (required, up to 2,000 characters)",
        "contact_submit": "Send",
        "contact_sent": "Your message has been sent. Thank you.",
        "contact_errors": {
            "name": "Please enter your name.",
            "affiliation": "Please enter your affiliation.",
            "email": "Please enter a valid email address.",
            "message": "Please enter a message.",
            "too_long": "Please keep the message within 2,000 characters.",
            "interval": "Please wait about a minute before sending another message.",
            "session_limit": "You can send up to 3 messages at a time. Please try again later.",
            "limit": "Today's limit for inquiries has been reached. Please try again tomorrow.",
            "unavailable": "Inquiries are not being accepted at the moment.",
            "failed": "Sending failed. Please try again later.",
        },
        # CSV一括入力
        "batch_intro": "Predict up to {max:,} samples at once from a CSV file. The first {n} results are shown on screen; download the CSV for all results.",
        "batch_columns": (
            "| Column | Content |\n| --- | --- |\n"
            "| TN_HU | Tumor nuclei (0–100) |\n"
            "| CancerType | Cancer type (a name from the list below) |\n"
            "| OpeBx | Ope (surgical) or Bx (biopsy) |\n"
            "| CollectionDate | Collection date (e.g. 2024/01/01) |\n"
            "| SubmissionDate | Submission date (e.g. 2025/04/05) |\n"
            "| StorageTime | Storage time in days; only when not using the two dates |\n\n"
            "Enter either the collection and submission dates or StorageTime. Rows with both use the dates. "
            "UTF-8 and Shift-JIS files are supported."
        ),
        "format": "CSV format (columns and cancer types)",
        "cancer_list": "Cancer types (names to use for CancerType)",
        "template": "Download the input template (CSV)",
        "upload": "Choose a CSV file",
        "batch_summary": "Predicted {ok:,} of {total:,} rows ({err:,} errors, {warn:,} warnings).",
        "batch_has_errors": "Rows with errors were not predicted. The reason is given in the Message column of the result CSV.",
        "preview": "First {n} results",
        "row": "Row",
        "message": "Message",
        "download": "Download results as CSV (all {total:,} rows)",
        "file_errors": {
            "empty": "The CSV contains no data.",
            "too_many": "Too many rows ({detail:,}). Please submit up to {max:,} rows at a time.",
            "missing_columns": "Required columns are missing: {detail}",
            "encoding": "Could not read the file encoding. Please save the CSV as UTF-8 or Shift-JIS.",
            "parse": "Could not read the file as CSV. Please check its format.",
        },
        "row_messages": {
            "missing_tn": "TN_HU is empty",
            "tn_not_number": "TN_HU is not a number",
            "tn_range": "TN_HU is outside 0–100",
            "missing_cancer": "CancerType is empty",
            "unknown_cancer": "CancerType is not in the list",
            "missing_opebx": "OpeBx is empty",
            "unknown_opebx": "OpeBx must be Ope or Bx",
            "missing_storage": "Collection/submission dates (or StorageTime) are empty",
            "bad_date": "Could not read the date (e.g. 2024/01/01)",
            "date_order": "The submission date is before the collection date",
            "storage_invalid": "StorageTime is not a number of 0 or more",
            "storage_range": "Warning: storage time is outside the training data range (0–{max:,} days)",
        },
    },
}

# がん種の日本語名（英語表示では学習時の名称をそのまま使う）
CANCER_JA = {
    "Adrenal_Gland": "副腎",
    "Ampulla_of_Vater": "ファーター乳頭部",
    "Biliary_Tract": "胆道",
    "Bladder_Urinary_Tract": "膀胱・尿路",
    "Bone": "骨",
    "Bowel": "腸",
    "Breast": "乳腺",
    "CNS_Brain": "中枢神経・脳",
    "Cervix": "子宮頸部",
    "Esophagus_Stomach": "食道・胃",
    "Eye": "眼",
    "Head_and_Neck": "頭頸部",
    "Kidney": "腎臓",
    "Liver": "肝臓",
    "Lung": "肺",
    "Other": "その他",
    "Ovary_Fallopian_Tube": "卵巣・卵管",
    "Pancreas": "膵臓",
    "Penis": "陰茎",
    "Peripheral_Nervous_System": "末梢神経",
    "Peritoneum": "腹膜",
    "Pleura": "胸膜",
    "Prostate": "前立腺",
    "Skin": "皮膚",
    "Soft_Tissue": "軟部組織",
    "Testis": "精巣",
    "Thymus": "胸腺",
    "Thyroid": "甲状腺",
    "Uterus": "子宮",
}

VERDICT_CLASS = {"Pass": "qc-pass", "Qualified": "qc-qualified"}

# 結果表のスタイル（ライト/ダーク両方で読めるよう半透明色を使う）
TABLE_CSS = """
<style>
.qc-table { width: 100%; border-collapse: collapse; margin: 0.25rem 0 0.75rem; }
.qc-table th { text-align: left; font-weight: 600; font-size: 0.85rem; opacity: 0.7;
  padding: 0.5rem 0.75rem; border-bottom: 1px solid rgba(128,128,128,0.35); }
.qc-table td { padding: 0.75rem; border-bottom: 1px solid rgba(128,128,128,0.2); vertical-align: middle; }
.qc-model { font-weight: 600; }
.qc-sub { display: block; font-size: 0.8rem; opacity: 0.65; }
.qc-badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
  font-weight: 600; font-size: 0.9rem; }
.qc-pass { color: #1a7f37; background: rgba(46,160,67,0.15); }
.qc-qualified { color: #b35900; background: rgba(230,126,34,0.18); }
.qc-prob { display: flex; align-items: center; gap: 0.6rem; }
.qc-bar { flex: 1; min-width: 48px; max-width: 160px; height: 8px; border-radius: 4px; background: rgba(128,128,128,0.2); }
.qc-bar > div { height: 100%; border-radius: 4px; background: #1f6feb; }
.qc-num { font-variant-numeric: tabular-nums; min-width: 3.5rem; text-align: right; }
.qc-legend { font-size: 0.9rem; line-height: 2; margin-bottom: 0.25rem; }
</style>
"""


def result_table(result, T) -> str:
    """4モデルの判定と確率をHTMLの表にする"""
    rows = []
    for r in result.itertuples():
        badge = f'<span class="qc-badge {VERDICT_CLASS.get(r.pred, "")}">{r.pred}</span>'
        if r.model == "SVM":
            prob = f'<span class="qc-num">{T["prob_na"]} {T["mark"]}</span>'
        else:
            pct = r.prob * 100
            prob = (
                f'<div class="qc-prob"><div class="qc-bar"><div style="width:{pct:.1f}%"></div></div>'
                f'<span class="qc-num">{pct:.1f}%</span></div>'
            )
        rows.append(
            f'<tr><td><span class="qc-model">{r.model}</span>'
            f'<span class="qc-sub">{T["models"][r.model]}</span></td>'
            f"<td>{badge}</td><td>{prob}</td></tr>"
        )
    return (
        f'<table class="qc-table"><thead><tr><th>{T["model"]}</th><th>{T["verdict"]}</th>'
        + f'<th>{T["prob"]}</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


@st.cache_resource(show_spinner=False)
def get_predictor() -> QCPredictor:
    """モデルはアプリ起動中に1回だけ読み込む"""
    return QCPredictor()


def default_lang() -> str:
    """ブラウザの言語設定が日本語なら日本語、それ以外は英語"""
    try:
        locale = (st.context.locale or "").lower()
    except Exception:
        locale = ""
    return "ja" if locale.startswith("ja") else "en"


def cancer_label(code: str, lang: str) -> str:
    if lang == "ja":
        return f"{CANCER_JA.get(code, code)}（{code}）"
    return code.replace("_", " ")


def row_message(codes, T) -> str:
    """行ごとの判定結果コードを文言にする"""
    msgs = T["row_messages"]
    return "; ".join(msgs[c].format(max=STORAGE_TRAIN_MAX) for c in codes)


def fmt_pred(pred, prob=None) -> str:
    """一括入力のプレビュー用：判定（＋確率）を1つの文字列にする"""
    if pred is None or (isinstance(pred, float) and pd.isna(pred)):
        return ""
    if prob is None or pd.isna(prob):
        return str(pred)
    return f"{pred} ({prob * 100:.1f}%)"


# ------------------------------------------------------------
# 画面
# ------------------------------------------------------------
st.set_page_config(page_title="QC Predictor", page_icon="🧪", layout="centered")
st.html(TABLE_CSS)  # 結果表・凡例のスタイル（画面全体で1回だけ読み込む）

if "lang" not in st.session_state:
    st.session_state.lang = default_lang()

head_l, head_r = st.columns([3, 1], vertical_alignment="bottom")
with head_r:
    st.segmented_control(
        "Language",
        options=["ja", "en"],
        format_func=lambda x: {"ja": "日本語", "en": "English"}[x],
        key="lang",
        label_visibility="collapsed",
    )
lang = st.session_state.lang or "ja"
T = TEXT[lang]

with head_l:
    st.title(T["title"])
st.write(T["lead"])
ref = f"[{REFERENCE}]({REFERENCE_URL})" if REFERENCE_URL else REFERENCE
st.caption(T["reference"].format(ref=ref) + "  \n" + T["cite"])
st.code(CITATION, language=None, wrap_lines=True)  # 右上のボタンでコピーできる

with st.spinner(T["loading"]):
    predictor = get_predictor()

# 「その他」は末尾に置く
cancer_options = [c for c in predictor.cancer_types if c != "Other"] + ["Other"]

# 入力方法の切り替え（言語を変えても選択が保たれるよう、値は言語に依存しないキーにする）
if "mode" not in st.session_state:
    st.session_state.mode = "single"
st.segmented_control(
    "Mode",
    options=["single", "batch"],
    format_func=lambda x: T["mode_single"] if x == "single" else T["mode_batch"],
    key="mode",
    label_visibility="collapsed",
)
mode = st.session_state.mode or "single"


def option_index(options, value):
    return options.index(value) if value in options else None


# ============================================================
# 1件ずつ入力
# ============================================================
def single_view():
    # 言語を切り替えるとフォームを作り直すため、最後に送信した値を初期値として引き継ぐ
    raw = st.session_state.get("raw_input", {})

    with st.form(f"qc_form_{lang}"):
        tn_hu = st.number_input(
            T["tn_hu"],
            min_value=0,
            max_value=100,
            value=raw.get("tn_hu"),
            step=1,
            help=T["tn_hu_help"],
            placeholder="0–100",
            key=f"tn_hu_{lang}",
        )
        cancer_type = st.selectbox(
            T["cancer_type"],
            options=cancer_options,
            index=option_index(cancer_options, raw.get("cancer_type")),
            format_func=lambda c: cancer_label(c, lang),
            placeholder=T["select"],
            key=f"cancer_type_{lang}",
        )
        ope_bx = st.radio(
            T["ope_bx"],
            options=["Ope", "Bx"],
            index=option_index(["Ope", "Bx"], raw.get("ope_bx")),
            format_func=lambda x: T["ope_bx_labels"][x],
            horizontal=True,
            key=f"ope_bx_{lang}",
        )
        date_args = dict(min_value=date(1980, 1, 1), max_value=date(2100, 12, 31), format="YYYY/MM/DD")
        c1, c2 = st.columns(2)
        with c1:
            collection = st.date_input(
                T["collection"], value=raw.get("collection"), key=f"collection_{lang}", **date_args
            )
        with c2:
            submission = st.date_input(
                T["submission"], value=raw.get("submission"), key=f"submission_{lang}", **date_args
            )
        st.caption(T["storage_help"])
        submitted = st.form_submit_button(T["submit"], type="primary", use_container_width=True)

    if submitted:
        # 入力チェック
        fields = {
            T["tn_hu"]: tn_hu,
            T["cancer_type"]: cancer_type,
            T["ope_bx"]: ope_bx,
            T["collection"]: collection,
            T["submission"]: submission,
        }
        st.session_state.raw_input = {
            "tn_hu": tn_hu,
            "cancer_type": cancer_type,
            "ope_bx": ope_bx,
            "collection": collection,
            "submission": submission,
        }
        missing = [name for name, v in fields.items() if v is None]
        st.session_state.pop("last_input", None)
        if missing:
            st.error(T["err_missing"] + ("、" if lang == "ja" else ", ").join(missing))
        elif storage_days(collection, submission) < 0:
            st.error(T["err_order"])
        else:
            # 言語を切り替えても結果を表示し続けるため、最後の入力を保持する
            st.session_state.last_input = {
                "tn_hu": tn_hu,
                "cancer_type": cancer_type,
                "ope_bx": ope_bx,
                "days": storage_days(collection, submission),
            }
            st.session_state.notify_pending = True  # 結果を表示するときに1回だけメール通知する

    last = st.session_state.get("last_input")
    if last:
        result = predictor.predict_one(last["tn_hu"], last["cancer_type"], last["ope_bx"], last["days"])
        if st.session_state.pop("notify_pending", False):
            notify.notify_single(last, result, lang, last["days"] > STORAGE_TRAIN_MAX)

        st.subheader(T["result"])
        st.markdown(T["storage_line"].format(days=last["days"]))
        if last["days"] > STORAGE_TRAIN_MAX:
            st.warning(T["warn_range"].format(max=STORAGE_TRAIN_MAX))
        st.caption(
            f"{T['summary']}TN_HU {last['tn_hu']}% / {cancer_label(last['cancer_type'], lang)}"
            f" / {T['ope_bx_labels'][last['ope_bx']]}"
        )
        st.markdown(result_table(result, T), unsafe_allow_html=True)
        legend()


def legend():
    st.markdown(
        f'<p class="qc-legend"><span class="qc-badge qc-pass">Pass</span> {T["legend_pass"]}'
        f'<br><span class="qc-badge qc-qualified">Qualified</span> {T["legend_qualified"]}</p>',
        unsafe_allow_html=True,
    )
    st.caption(T["svm_note"])


# ============================================================
# CSVで一括入力
# ============================================================
def batch_view():
    st.write(T["batch_intro"].format(max=MAX_BATCH_ROWS, n=PREVIEW_ROWS))
    with st.expander(T["format"]):
        st.markdown(T["batch_columns"])
        st.markdown(f"**{T['cancer_list']}**")
        st.markdown(
            "\n".join(f"- `{c}`" + (f"　{CANCER_JA[c]}" if lang == "ja" else "") for c in cancer_options)
        )
    st.download_button(
        T["template"],
        data=TEMPLATE_CSV.encode("utf-8-sig"),
        file_name="qc_input_template.csv",
        mime="text/csv",
    )

    # ラベルは言語で変えない（変えるとアップロード済みファイルが消えるため）
    st.markdown(f"**{T['upload']}**")
    uploaded = st.file_uploader("CSV", type=["csv"], key="batch_file", label_visibility="collapsed")
    if uploaded is None:
        st.session_state.pop("batch", None)
        return

    # 同じファイルは再計算しない（ダウンロードや言語切替のたびに画面が再実行されるため）
    cache = st.session_state.get("batch")
    if cache is None or cache["file_id"] != uploaded.file_id:
        try:
            raw = read_csv_bytes(uploaded.getvalue())
            cache = {"file_id": uploaded.file_id, "result": predict_batch(predictor, raw), "error": None}
        except BatchError as e:
            cache = {"file_id": uploaded.file_id, "result": None, "error": (e.code, e.detail)}
        except Exception:
            cache = {"file_id": uploaded.file_id, "result": None, "error": ("parse", None)}
        st.session_state.batch = cache
        if cache["result"] is not None:
            codes = cache["result"]["codes"]
            n_err = int(codes.apply(lambda c: any(x in ERROR_CODES for x in c)).sum())
            notify.notify_batch(
                total=len(codes),
                ok=len(codes) - n_err,
                errors=n_err,
                warnings=int(codes.apply(lambda c: "storage_range" in c).sum()),
                lang=lang,
            )

    if cache["error"]:
        code, detail = cache["error"]
        if isinstance(detail, list):
            detail = ", ".join(detail)
        st.error(T["file_errors"][code].format(detail=detail, max=MAX_BATCH_ROWS))
        return

    res = cache["result"]
    is_err = res["codes"].apply(lambda c: any(x in ERROR_CODES for x in c))
    has_warn = res["codes"].apply(lambda c: "storage_range" in c)
    total, n_err = len(res), int(is_err.sum())

    st.subheader(T["result"])
    st.write(T["batch_summary"].format(total=total, ok=total - n_err, err=n_err, warn=int(has_warn.sum())))
    if n_err:
        st.warning(T["batch_has_errors"])

    # 先頭5件のプレビュー
    st.markdown(f"**{T['preview'].format(n=min(PREVIEW_ROWS, total))}**")
    head = res.head(PREVIEW_ROWS)
    preview = pd.DataFrame(
        {
            T["row"]: range(1, len(head) + 1),
            "StorageTime": head["StorageTime"].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}"),
            **{
                m: [
                    fmt_pred(p, head[f"{m}_prob_Qualified"].iloc[i] if m != "SVM" else None)
                    for i, p in enumerate(head[f"{m}_pred"])
                ]
                for m in MODEL_ORDER
            },
            T["message"]: head["codes"].map(lambda c: row_message(c, T)),
        }
    )
    st.dataframe(preview, hide_index=True, use_container_width=True)
    legend()

    # 全件の結果CSV（Excelで文字化けしないよう BOM付きUTF-8）
    out = res.drop(columns=["codes"]).copy()
    out["Message"] = res["codes"].map(lambda c: row_message(c, T))
    stem = Path(uploaded.name).stem
    st.download_button(
        T["download"].format(total=total),
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{stem}_qc_result.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# お問い合わせ
# ============================================================
def contact_view():
    with st.expander(T["contact"]):
        if st.session_state.pop("contact_done", False):
            st.success(T["contact_sent"])
        st.caption(T["contact_intro"])
        # 送信に成功したらフォームを空にするため、キーに番号を付けて作り直す
        n = st.session_state.get("contact_n", 0)
        with st.form(f"contact_form_{lang}_{n}"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input(T["contact_name"], max_chars=100)
            with c2:
                affiliation = st.text_input(T["contact_affiliation"], max_chars=200)
            email = st.text_input(T["contact_email"], max_chars=254)
            message = st.text_area(T["contact_message"], height=160)
            submitted = st.form_submit_button(T["contact_submit"])

        if not submitted:
            return
        E = T["contact_errors"]
        one_line = lambda v: " ".join(v.split())  # 改行や余分な空白を除く（メールの件名に使うため）
        name, affiliation = one_line(name), one_line(affiliation)
        email, message = email.strip(), message.strip()
        sent_times = st.session_state.setdefault("contact_times", [])
        if not name:
            st.error(E["name"])
        elif not affiliation:
            st.error(E["affiliation"])
        elif not notify.is_valid_email(email):
            st.error(E["email"])
        elif not message:
            st.error(E["message"])
        elif len(message) > CONTACT_MAX_CHARS:
            st.error(E["too_long"])
        elif len(sent_times) >= CONTACT_PER_SESSION:
            st.error(E["session_limit"])
        elif sent_times and time.time() - sent_times[-1] < CONTACT_INTERVAL_SEC:
            st.error(E["interval"])
        else:
            status = notify.send_contact(name, affiliation, email, message, lang)
            if status == "sent":
                sent_times.append(time.time())
                st.session_state.contact_done = True
                st.session_state.contact_n = n + 1
                st.rerun()
            else:
                st.error(E[status])


if mode == "batch":
    batch_view()
else:
    single_view()

st.divider()
st.caption(T["disclaimer"])
st.caption(f"{T['license']}　[{T['source']}]({REPO_URL})")
contact_view()
