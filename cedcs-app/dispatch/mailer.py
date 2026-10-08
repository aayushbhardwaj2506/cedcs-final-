"""Outgoing mail with two modes and a hard safety net.

  DISPATCH_MODE=outbox (default)  Nothing is sent. Every message is recorded and shown in the app's outbox ("dry run").
  DISPATCH_MODE=live (or smtp)    Real delivery. MAIL_PROVIDER picks how:
      smtp (default)  SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / SMTP_FROM        (works locally; hosts such as Render's
                      free plan block outgoing SMTP)
      brevo           BREVO_API_KEY + MAIL_FROM (a sender verified in Brevo)               (HTTPS API, works anywhere)
      resend          RESEND_API_KEY + MAIL_FROM (onboarding@resend.dev, or your domain)   (HTTPS API, works anywhere)

Safety rules that hold in every mode:
  * DISPATCH_TEST_INBOX redirects EVERY message to that one address (sandbox), with the intended recipient shown in the
    subject and body, so real delivery can be tried without contacting anyone else.
  * The synthetic hospital and ambulance addresses use reserved domains (example.org); in live mode they are refused unless
    the sandbox inbox is set, so a fictional hospital can never cause a bounce storm or reach a stranger.
  * Credentials come from the environment only and never appear in logs, errors or API responses.
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
import time
from email.message import EmailMessage
from email.utils import formatdate, make_msgid, parseaddr
from typing import Optional

import requests

logger = logging.getLogger("cedcs.mailer")

_RESERVED_SUFFIXES = (".example", ".invalid", ".test", ".localhost", "example.org", "example.com", "example.net")
_PROVIDER_LABEL = {"smtp": "SMTP", "brevo": "Brevo", "resend": "Resend"}
BREVO_URL = "https://api.brevo.com/v3/smtp/email"
RESEND_URL = "https://api.resend.com/emails"


def mode() -> str:
    """"smtp" means live delivery by any transport (the name is kept for the UI and the stored dispatch records)."""
    return "smtp" if os.environ.get("DISPATCH_MODE", "outbox").strip().lower() in ("smtp", "live") else "outbox"


def provider() -> str:
    p = os.environ.get("MAIL_PROVIDER", "smtp").strip().lower()
    return p if p in _PROVIDER_LABEL else "smtp"


def sandbox_inbox() -> Optional[str]:
    v = os.environ.get("DISPATCH_TEST_INBOX", "").strip()
    return v or None


def sender() -> str:
    return (os.environ.get("MAIL_FROM") or os.environ.get("SMTP_FROM") or "").strip()


def configured() -> bool:
    p = provider()
    if p == "brevo":
        return bool(os.environ.get("BREVO_API_KEY") and sender())
    if p == "resend":
        return bool(os.environ.get("RESEND_API_KEY") and sender())
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")) and bool(sender())


def smtp_configured() -> bool:  # kept for older callers
    return configured()


def is_synthetic(address: str) -> bool:
    domain = (address or "").rsplit("@", 1)[-1].lower()
    return any(domain == s.lstrip(".") or domain.endswith(s) for s in _RESERVED_SUFFIXES)


def describe() -> dict:
    """Safe-to-show summary (no secrets) of how mail will be handled right now."""
    m, via = mode(), _PROVIDER_LABEL[provider()]
    if m == "outbox":
        label = "DRY RUN: nothing leaves this computer"
    elif not configured():
        label = f"LIVE mode selected but {via} is not configured: sending will fail"
    else:
        label = f"LIVE via {via}, redirected to the sandbox inbox" if sandbox_inbox() else f"LIVE via {via}"
    return {"mode": m, "provider": provider(), "sandbox_inbox": bool(sandbox_inbox()), "smtp_configured": configured(), "configured": configured(), "label": label}


def _secrets() -> list:
    return [v for k in ("SMTP_PASSWORD", "BREVO_API_KEY", "RESEND_API_KEY") if (v := os.environ.get(k))]


def _scrub(text: str) -> str:
    for s in _secrets():
        text = text.replace(s, "***")
    return text


def _send_smtp(actual: str, subject: str, text: str, html: str) -> None:
    em = EmailMessage()
    em["From"], em["To"], em["Subject"] = sender(), actual, subject
    em["Date"], em["Message-ID"] = formatdate(localtime=True), make_msgid(domain="cedcs.local")
    em.set_content(text)
    if html:
        em.add_alternative(html, subtype="html")
    port = int(os.environ.get("SMTP_PORT", "587"))
    if port == 465:
        server = smtplib.SMTP_SSL(os.environ["SMTP_HOST"], port, timeout=15, context=ssl.create_default_context())
    else:
        server = smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=15)
        server.starttls(context=ssl.create_default_context())
    with server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
        server.send_message(em)


def _check(resp) -> None:
    if resp.status_code < 300:
        return
    low = resp.text.lower()
    hint = ""
    if resp.status_code == 401:
        hint = " -> the provider does not recognise the API key. Brevo needs an API key (starts with xkeysib-), not the SMTP key; check BREVO_API_KEY for spaces or a missing part."
    elif resp.status_code == 403:
        hint = " -> access denied: check the key's permissions and any IP restrictions on it."
    elif resp.status_code == 400 and "sender" in low:
        hint = " -> the sender address is not verified with the provider: verify MAIL_FROM there first."
    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160].strip()}{hint}")


def _send_brevo(actual: str, subject: str, text: str, html: str) -> None:
    name, addr = parseaddr(sender())
    body = {"sender": {"email": addr, **({"name": name} if name else {})}, "to": [{"email": actual}], "subject": subject, "textContent": text}
    if html:
        body["htmlContent"] = html
    _check(requests.post(BREVO_URL, json=body, headers={"api-key": os.environ["BREVO_API_KEY"], "accept": "application/json"}, timeout=20))


def _send_resend(actual: str, subject: str, text: str, html: str) -> None:
    body = {"from": sender(), "to": [actual], "subject": subject, "text": text}
    if html:
        body["html"] = html
    _check(requests.post(RESEND_URL, json=body, headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}"}, timeout=20))


_TRANSPORTS = {"smtp": _send_smtp, "brevo": _send_brevo, "resend": _send_resend}


def send(msg: dict) -> dict:
    """msg: {to_email, subject, body_text, body_html}. Returns {status, actual_to, subject, error} (never raises)."""
    intended = msg["to_email"]
    if mode() == "outbox":
        return {"status": "LOGGED", "actual_to": intended, "subject": msg["subject"], "error": None}

    if not configured():
        need = {"brevo": "BREVO_API_KEY and MAIL_FROM", "resend": "RESEND_API_KEY and MAIL_FROM"}.get(provider(), "SMTP_HOST/USER/PASSWORD/FROM")
        return {"status": "FAILED", "actual_to": intended, "subject": msg["subject"], "error": f"{_PROVIDER_LABEL[provider()]} is not configured ({need})"}
    box = sandbox_inbox()
    if box is None and is_synthetic(intended):
        return {"status": "FAILED", "actual_to": intended, "subject": msg["subject"],
                "error": "synthetic address on a reserved domain; set DISPATCH_TEST_INBOX to receive test mail"}

    actual = box or intended
    subject = f"[SANDBOX for {intended}] {msg['subject']}" if box else msg["subject"]
    text = (f"*** SANDBOX: this message was addressed to {intended} and redirected to you ***\n\n" if box else "") + msg["body_text"]
    try:
        _TRANSPORTS[provider()](actual, subject, text, msg.get("body_html") or "")
    except Exception as exc:  # network, auth, refused recipient...: report the type only, never the credentials
        logger.warning("%s send failed: %s", provider(), type(exc).__name__)
        return {"status": "FAILED", "actual_to": actual, "subject": subject, "error": _scrub(f"{type(exc).__name__}: {str(exc)[:160]}")}
    return {"status": "SENT", "actual_to": actual, "subject": subject, "error": None}


def now() -> float:
    return time.time()
