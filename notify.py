"""
解析実行時のメール通知（ツールが正常に動作しているかの確認用）と、お問い合わせの送信

送信設定は Streamlit の Secrets に置く（GitHub には載せない）。
Secrets に [email] がなければ何もしない（ローカル実行時など）。

    [email]
    sender = "送信元のGmailアドレス"
    app_password = "Gmailのアプリパスワード（16文字）"
    to = "通知を受け取るアドレス"
    # 以下は省略可（既定値は Gmail）
    # host = "smtp.gmail.com"
    # port = 465
    # use_ssl = true
"""
from __future__ import annotations

import logging
import re
import smtplib
import ssl
import threading
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import streamlit as st

log = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
# 1日に送る上限（Gmailの送信上限500通/日より十分少なくする）。
# 解析通知が上限に達してもお問い合わせは届くよう、別々に数える
DAILY_LIMIT = {"notify": 200, "contact": 50}
SUBJECT_PREFIX = "[QC Predictor]"


@st.cache_resource
def _counter() -> dict:
    """アプリ全体で共有する送信数カウンター（日付が変わったらリセット）"""
    return {"day": None, "count": {}, "lock": threading.Lock()}


def _config():
    try:
        cfg = st.secrets["email"]
        return {
            "sender": cfg["sender"],
            "password": cfg.get("app_password", ""),
            "to": cfg["to"],
            "host": cfg.get("host", "smtp.gmail.com"),
            "port": int(cfg.get("port", 465)),
            "use_ssl": bool(cfg.get("use_ssl", True)),
        }
    except Exception:
        return None  # Secrets 未設定 → 通知しない


def _within_limit(kind: str = "notify") -> bool:
    c = _counter()
    today = datetime.now(JST).date()
    with c["lock"]:
        if c["day"] != today:
            c["day"], c["count"] = today, {}
        if c["count"].get(kind, 0) >= DAILY_LIMIT[kind]:
            return False
        c["count"][kind] = c["count"].get(kind, 0) + 1
        return True


def _send(cfg: dict, subject: str, body: str, reply_to: str | None = None) -> bool:
    msg = EmailMessage()
    msg["Subject"] = f"{SUBJECT_PREFIX} {subject}"
    msg["From"] = cfg["sender"]
    msg["To"] = cfg["to"]
    if reply_to:
        msg["Reply-To"] = reply_to  # 届いたメールに「返信」すると問い合わせた人に届く
    msg.set_content(body)
    try:
        if cfg["use_ssl"]:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context(), timeout=15) as s:
                s.login(cfg["sender"], cfg["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=15) as s:
                if cfg["password"]:
                    s.login(cfg["sender"], cfg["password"])
                s.send_message(msg)
        return True
    except Exception:
        # 通知の失敗で画面を止めない。原因は Streamlit Cloud のログで確認できる
        log.exception("Failed to send notification email")
        return False


def send_async(subject: str, body: str) -> bool:
    """
    別スレッドでメールを送る（画面の表示を待たせない）。
    設定がない・上限超過のときは送らずに False を返す。
    """
    cfg = _config()
    if cfg is None or not _within_limit():
        return False
    threading.Thread(target=_send, args=(cfg, subject, body), daemon=True).start()
    return True


def _now() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")


def notify_single(inputs: dict, result, lang: str, out_of_range: bool) -> bool:
    """1件ずつ入力：入力内容（日付は送らず保存期間の日数のみ）と判定結果"""
    lines = [
        "1件ずつ入力で解析が実行されました。",
        "",
        f"日時: {_now()}",
        f"表示言語: {lang}",
        "",
        "■ 入力内容",
        f"TN_HU（腫瘍細胞含有率）: {inputs['tn_hu']}%",
        f"CancerType（がん種）: {inputs['cancer_type']}",
        f"OpeBx（検体の種類）: {inputs['ope_bx']}",
        f"StorageTime（保存期間）: {inputs['days']:,}日" + ("（学習データの範囲外）" if out_of_range else ""),
        "",
        "■ 判定結果（Qualified確率）",
    ]
    for r in result.itertuples():
        prob = "―" if r.model == "SVM" else f"{r.prob * 100:.1f}%"
        lines.append(f"{r.model}: {r.pred}（{prob}）")
    return send_async("1件の解析が実行されました", "\n".join(lines))


def notify_batch(total: int, ok: int, errors: int, warnings: int, lang: str) -> bool:
    """CSV一括入力：解析があったことと件数のみ（入力内容・結果は送らない）"""
    body = "\n".join(
        [
            "CSV一括入力で解析が実行されました。",
            "",
            f"日時: {_now()}",
            f"表示言語: {lang}",
            f"件数: {total:,}件（予測 {ok:,}件、エラー {errors:,}件、警告 {warnings:,}件）",
            "",
            "※ CSVの入力内容と判定結果はメールに含めていません。",
        ]
    )
    return send_async(f"CSV一括解析が実行されました（{total:,}件）", body)


# ============================================================
# お問い合わせ
# ============================================================
EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def is_valid_email(address: str) -> bool:
    return bool(EMAIL_RE.match(address or ""))


def contact_available() -> bool:
    """送信設定（Secrets）があるか"""
    return _config() is not None


def send_contact(name: str, affiliation: str, email: str, message: str, lang: str) -> str:
    """
    お問い合わせを管理者に送る（結果を画面に出すため、送り終わるまで待つ）。
    戻り値："sent" / "unavailable"（設定なし）/ "limit"（本日の上限）/ "failed"（送信失敗）
    """
    cfg = _config()
    if cfg is None:
        return "unavailable"
    if not _within_limit("contact"):
        return "limit"
    body = "\n".join(
        [
            "QC予測ツールからお問い合わせがありました。",
            "このメールに返信すると、問い合わせた方に届きます。",
            "",
            f"日時: {_now()}",
            f"表示言語: {lang}",
            f"お名前: {name}",
            f"所属: {affiliation}",
            f"メールアドレス: {email}",
            "",
            "■ お問い合わせ内容",
            message,
        ]
    )
    ok = _send(cfg, f"お問い合わせ（{name}／{affiliation}）", body, reply_to=email)
    return "sent" if ok else "failed"
