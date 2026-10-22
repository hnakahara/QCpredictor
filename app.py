"""
QC予測 Webアプリ（Streamlit）

入力5項目（TN_HU・がん種・手術/生検・検体取得日・検体提出日）から
保存期間を算出し、4モデル（GBM / RF / SVM / NN）のQC判定を表示する。
"""
import os
from datetime import date

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TensorFlowのログを抑制

import streamlit as st

from qc_core import QCPredictor, storage_days

REPO_URL = "https://github.com/hnakahara/QCpredictor"

# ------------------------------------------------------------
# 表示文言（日本語 / 英語）
# ------------------------------------------------------------
TEXT = {
    "ja": {
        "page_title": "QC予測ツール",
        "title": "QC予測ツール",
        "lead": "検体の情報を入力すると、4つの機械学習モデルがQCチェックの見込みを判定します。",
        "tn_hu": "腫瘍細胞含有率（%）",
        "tn_hu_help": "病理医が評価した腫瘍細胞含有率（0〜100）",
        "cancer_type": "がん種",
        "ope_bx": "検体の種類",
        "collection": "検体取得日",
        "submission": "検体提出日",
        "storage_help": "保存期間は「提出日 − 取得日」の日数で自動計算します。",
        "select": "選択してください",
        "submit": "予測する",
        "loading": "モデルを読み込んでいます（初回は数十秒かかることがあります）…",
        "err_missing": "未入力の項目があります：",
        "err_order": "検体提出日は検体取得日と同じ日か、それ以降にしてください。",
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
        "disclaimer": "本ツールの結果は研究用途の参考値です。診療上の判断には使用しないでください。入力内容はサーバーに保存されません。",
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
    },
    "en": {
        "page_title": "QC Predictor",
        "title": "QC Predictor",
        "lead": "Enter the sample information and four machine learning models will predict its QC outcome.",
        "tn_hu": "Tumor nuclei (%)",
        "tn_hu_help": "Percent tumor nuclei assessed by a pathologist (0–100)",
        "cancer_type": "Cancer type",
        "ope_bx": "Sample type",
        "collection": "Collection date",
        "submission": "Submission date",
        "storage_help": "Storage time is calculated automatically as submission date − collection date, in days.",
        "select": "Select",
        "submit": "Predict",
        "loading": "Loading models (the first run can take a few tens of seconds)…",
        "err_missing": "Please fill in: ",
        "err_order": "The submission date must be on or after the collection date.",
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
        "disclaimer": "Results are for research reference only and must not be used for clinical decisions. Inputs are not stored on the server.",
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
        TABLE_CSS
        + f'<table class="qc-table"><thead><tr><th>{T["model"]}</th><th>{T["verdict"]}</th>'
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


# ------------------------------------------------------------
# 画面
# ------------------------------------------------------------
st.set_page_config(page_title="QC Predictor", page_icon="🧪", layout="centered")

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

with st.spinner(T["loading"]):
    predictor = get_predictor()

# 「その他」は末尾に置く
cancer_options = [c for c in predictor.cancer_types if c != "Other"] + ["Other"]

# 言語を切り替えるとフォームを作り直すため、最後に送信した値を初期値として引き継ぐ
raw = st.session_state.get("raw_input", {})


def option_index(options, value):
    return options.index(value) if value in options else None


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

last = st.session_state.get("last_input")
if last:
    result = predictor.predict_one(last["tn_hu"], last["cancer_type"], last["ope_bx"], last["days"])

    st.subheader(T["result"])
    st.markdown(T["storage_line"].format(days=last["days"]))
    st.caption(
        f"{T['summary']}TN_HU {last['tn_hu']}% / {cancer_label(last['cancer_type'], lang)}"
        f" / {T['ope_bx_labels'][last['ope_bx']]}"
    )
    st.markdown(result_table(result, T), unsafe_allow_html=True)
    st.markdown(
        f'<p class="qc-legend"><span class="qc-badge qc-pass">Pass</span> {T["legend_pass"]}'
        f'<br><span class="qc-badge qc-qualified">Qualified</span> {T["legend_qualified"]}</p>',
        unsafe_allow_html=True,
    )
    st.caption(T["svm_note"])

st.divider()
st.caption(T["disclaimer"])
st.caption(f"{T['license']}　[{T['source']}]({REPO_URL})")
