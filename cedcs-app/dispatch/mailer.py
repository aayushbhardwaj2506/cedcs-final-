"""Outgoing mail with two modes and a hard safety net.

  DISPATCH_MODE=outbox (default)  Nothing is sent. Every message is recorded and shown in the app's outbox ("dry run").
  DISPATCH_MODE=smtp              Real delivery through SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / SMTP_FROM.

Safety rules that hold in every mode:
  * DISPATCH_TEST_INBOX redirects EVERY message to that one address (sandbox), with the intended recipient shown in the
    subject and body, so real delivery can be tried without contacting anyone else.
  * The synthetic hospital and ambulance addresses use reserved domains (example.org); in smtp mode they are refused unless
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
from email.utils import formatdate, make_msgid
from typing import Optional

logger = logging.getLogger("cedcs.mailer")

_RESERVED_SUFFIXES = (".example", ".invalid", ".test", ".localhost", "example.org", "example.com", "example.net")


def mode() -> str:
    return "smtp" if os.environ.get("DISPATCH_MODE", "outbox").strip().lower() == "smtp" else "outbox"


def sandbox_inbox() -> Optional[str]:
    v = os.environ.get("DISPATCH_TEST_INBOX", "").strip()
    return v or None


def smtp_configured() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM"))


def is_synthetic(address: str) -> bool:
    domain = (address or "").rsplit("@", 1)[-1].lower()
    return any(domain == s.lstrip(".") or domain.endswith(s) for s in _RESERVED_SUFFIXES)


def describe() -> dict:
    """Safe-to-show summary (no secrets) of how mail will be handled right now."""
    m = mode()
    return {"mode": m, "sandbox_inbox": bool(sandbox_inbox()), "smtp_configured": smtp_configured(),
            "label": ("DRY RUN: nothing leaves this computer" if m == "outbox"
                      else ("LIVE via SMTP, redirected to the sandbox inbox" if sandbox_inbox() else "LIVE via SMTP")) if m == "outbox" or smtp_configured()
            else "LIVE mode selected but SMTP is not configured: sending will fail"}


def send(msg: dict) -> dict:
    """msg: {to_email, subject, body_text, body_html}. Returns {status, actual_to, subject, error} (never raises)."""
    intended = msg["to_email"]
    if mode() == "outbox":
        return {"status": "LOGGED", "actual_to": intended, "subject": msg["subject"], "error": None}

    if not smtp_configured():
        return {"status": "FAILED", "actual_to": intended, "subject": msg["subject"], "error": "SMTP is not configured (SMTP_HOST/USER/PASSWORD/FROM)"}
    box = sandbox_inbox()
    if box is None and is_synthetic(intended):
        return {"status": "FAILED", "actual_to": intended, "subject": msg["subject"],
                "error": "synthetic address on a reserved domain; set DISPATCH_TEST_INBOX to receive test mail"}

    actual = box or intended
    subject = f"[SANDBOX for {intended}] {msg['subject']}" if box else msg["subject"]
    text = (f"*** SANDBOX: this message was addressed to {intended} and redirected to you ***\n\n" if box else "") + msg["body_text"]
    em = EmailMessage()
    em["From"], em["To"], em["Subject"] = os.environ["SMTP_FROM"], actual, subject
    em["Date"], em["Message-ID"] = formatdate(localtime=True), make_msgid(domain="cedcs.local")
    em.set_content(text)
    if msg.get("body_html"):
        em.add_alternative(msg["body_html"], subtype="html")
    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
        if port == 465:
            server = smtplib.SMTP_SSL(os.environ["SMTP_HOST"], port, timeout=15, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(os.environ["SMTP_HOST"], port, timeout=15)
            server.starttls(context=ssl.create_default_context())
        with server:
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            server.send_message(em)
    except Exception as exc:  # network, auth, refused recipient...: report the type only, never the credentials
        logger.warning("smtp send failed: %s", type(exc).__name__)
        return {"status": "FAILED", "actual_to": actual, "subject": subject, "error": f"{type(exc).__name__}: {str(exc)[:120]}".replace(os.environ["SMTP_PASSWORD"], "***")}
    return {"status": "SENT", "actual_to": actual, "subject": subject, "error": None}


def now() -> float:
    return time.time()
