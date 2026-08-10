"""
Email delivery for offer letters (and anything else that needs to email a
PDF). Three sending methods are supported, tried in this order:

1. GOOGLE APPS SCRIPT relay (use this if you specifically want Gmail to be
   the one sending) — a tiny script deployed on script.google.com sends via
   MailApp.sendEmail() using YOUR Gmail account, entirely inside Google's
   own servers. Your app just POSTs to the script's webhook URL over HTTPS
   (port 443), so campus/corporate WiFi blocking SMTP ports (587/465)
   doesn't matter — there's no SMTP connection at all. See setup steps in
   the docstring of send_via_google_apps_script() below.

2. RESEND API — also HTTPS-based, but sends from Resend's infrastructure
   rather than your own Gmail. Simpler to set up (just an API key, no script
   to deploy) if you don't care which address it's "from".
   Sign up free at https://resend.com (100 emails/day free, no credit card).

3. SMTP (fallback) — via smtplib, direct to Gmail/Outlook/Yahoo/custom.
   This is the one that gets blocked by networks that block outbound SMTP
   ports — kept as a fallback for networks where it does work.

Whichever secrets are present determines what's used, in the priority order
above. Credentials are intentionally NEVER stored in local_settings.json
(a plain JSON file on disk) — only read from st.secrets/.streamlit/secrets.toml
or environment variables, same pattern ai_engine.py uses for AI provider keys.

Example .streamlit/secrets.toml (Google Apps Script — real Gmail sending):
  GAS_WEBHOOK_URL = "https://script.google.com/macros/s/XXXXX/exec"
  GAS_SECRET = "some-random-string-you-picked"   # shared secret, checked by the script

Example .streamlit/secrets.toml (Resend):
  RESEND_API_KEY = "re_xxxxxxxxxxxx"
  RESEND_FROM_EMAIL = "onboarding@resend.dev"   # or your own verified domain

Example .streamlit/secrets.toml (SMTP):
  SMTP_EMAIL = "hr@yourcompany.com"
  SMTP_APP_PASSWORD = "your-16-char-app-password"
"""

import os
import base64
import smtplib
import requests
import streamlit as st
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

SMTP_PRESETS = {
    "Gmail": ("smtp.gmail.com", 587),
    "Outlook / Office 365": ("smtp.office365.com", 587),
    "Yahoo": ("smtp.mail.yahoo.com", 587),
}


def _get_secret(name: str) -> str | None:
    # Same lookup order as ai_engine._get_secret: session override (for
    # quick testing) -> st.secrets -> environment variable.
    try:
        override = st.session_state.get(f"override_{name}")
        if override:
            return override
    except Exception:
        pass
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def get_sender_email() -> str | None:
    return _get_secret("SMTP_EMAIL") or _get_secret("RESEND_FROM_EMAIL")


def is_gas_configured() -> bool:
    return bool(_get_secret("GAS_WEBHOOK_URL"))


def is_resend_configured() -> bool:
    return bool(_get_secret("RESEND_API_KEY"))


def is_smtp_configured() -> bool:
    return bool(_get_secret("SMTP_EMAIL") and _get_secret("SMTP_APP_PASSWORD"))


def is_configured() -> bool:
    return is_gas_configured() or is_resend_configured() or is_smtp_configured()


def _send_via_google_apps_script(to_email: str, subject: str, body_text: str,
                                  pdf_bytes: bytes, pdf_filename: str) -> tuple[bool, str]:
    """
    Sends via a Google Apps Script web app acting as a relay — the script
    runs inside Google's servers and calls MailApp.sendEmail() using YOUR
    Gmail account, so the email genuinely comes from Gmail. Your app only
    makes a plain HTTPS POST to reach it, so SMTP port blocking (common on
    campus/corporate WiFi) is irrelevant here.

    ONE-TIME SETUP (about 5 minutes):
    1. Go to https://script.google.com -> New project.
    2. Delete the default code and paste this in:

        function doPost(e) {
          var data = JSON.parse(e.postData.contents);
          if (data.secret !== "YOUR_SECRET_HERE") {
            return ContentService.createTextOutput(JSON.stringify({error: "bad secret"}));
          }
          var pdfBlob = Utilities.newBlob(
            Utilities.base64Decode(data.pdf_base64), "application/pdf", data.filename
          );
          MailApp.sendEmail({
            to: data.to,
            subject: data.subject,
            body: data.body,
            attachments: [pdfBlob]
          });
          return ContentService.createTextOutput(JSON.stringify({ok: true}));
        }

    3. Replace "YOUR_SECRET_HERE" with any random string you make up — this
       stops strangers from finding your URL and using it to spam people.
    4. Click Deploy -> New deployment -> type: Web app.
       - Execute as: Me
       - Who has access: Anyone
    5. Click Deploy, authorize with your Google account when prompted
       (you'll see an "unverified app" warning — that's expected for your
       own personal script; click Advanced -> Go to project (unsafe)).
    6. Copy the Web app URL it gives you (ends in /exec).
    7. Add to .streamlit/secrets.toml:
         GAS_WEBHOOK_URL = "https://script.google.com/macros/s/XXXXX/exec"
         GAS_SECRET = "the-same-random-string-from-step-3"
    """
    webhook_url = _get_secret("GAS_WEBHOOK_URL")
    secret = _get_secret("GAS_SECRET") or ""

    payload = {
        "to": to_email, "subject": subject, "body": body_text,
        "filename": pdf_filename,
        "pdf_base64": base64.b64encode(pdf_bytes).decode("ascii"),
        "secret": secret,
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=25)
        try:
            data = resp.json()
        except Exception:
            data = None
        if resp.status_code == 200 and data and data.get("ok"):
            return True, f"Sent to {to_email} via Gmail."
        error_detail = (data or {}).get("error") if data else resp.text
        return False, f"Google Apps Script returned an error: {error_detail or f'HTTP {resp.status_code}'}"
    except requests.exceptions.Timeout:
        return False, "The Apps Script webhook timed out — check the URL is correct and the script is deployed."
    except requests.exceptions.ConnectionError as e:
        return False, f"Couldn't reach the Apps Script webhook — check your internet connection. ({e})"
    except Exception as e:
        return False, f"Couldn't send via Google Apps Script: {e}"


def _send_via_resend(to_email: str, subject: str, body_text: str,
                      pdf_bytes: bytes, pdf_filename: str) -> tuple[bool, str]:
    """Sends via the Resend HTTPS API — no SMTP ports involved at all, so
    this works even on networks that block outbound SMTP entirely."""
    api_key = _get_secret("RESEND_API_KEY")
    from_email = _get_secret("RESEND_FROM_EMAIL") or "onboarding@resend.dev"

    payload = {
        "from": from_email,
        "to": [to_email],
        "subject": subject,
        "text": body_text,
        "attachments": [{
            "filename": pdf_filename,
            "content": base64.b64encode(pdf_bytes).decode("ascii"),
        }],
    }
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=20,
        )
        if resp.status_code in (200, 201):
            return True, f"Sent to {to_email}."
        try:
            detail = resp.json().get("message", resp.text)
        except Exception:
            detail = resp.text
        return False, f"Resend API error ({resp.status_code}): {detail}"
    except requests.exceptions.Timeout:
        return False, "Resend API request timed out — check your internet connection."
    except requests.exceptions.ConnectionError as e:
        return False, f"Couldn't reach the Resend API — check your internet connection. ({e})"
    except Exception as e:
        return False, f"Couldn't send via Resend: {e}"


def _resolve_host_port(provider: str) -> tuple[str, int]:
    if provider in SMTP_PRESETS:
        return SMTP_PRESETS[provider]
    host = _get_secret("SMTP_HOST") or ""
    port = int(_get_secret("SMTP_PORT") or 587)
    return host, port


SSL_PORTS = {
    "Gmail": 465,
    "Outlook / Office 365": 587,  # Outlook doesn't support SSL-on-connect the same way; STARTTLS only
    "Yahoo": 465,
}


def _build_message(sender_email, to_email, subject, body_text, pdf_bytes, pdf_filename):
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain"))

    part = MIMEBase("application", "octet-stream")
    part.set_payload(pdf_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{pdf_filename}"')
    msg.attach(part)
    return msg


def send_plain_email(to_email: str, subject: str, body_text: str) -> tuple[bool, str]:
    """
    Sends a plain text email with no attachment — used for things like
    verification codes rather than documents. Same priority as
    send_email_with_pdf: Google Apps Script (real Gmail) -> Resend -> SMTP.
    """
    if is_gas_configured():
        webhook_url = _get_secret("GAS_WEBHOOK_URL")
        secret = _get_secret("GAS_SECRET") or ""
        payload = {"to": to_email, "subject": subject, "body": body_text, "filename": "", "pdf_base64": "", "secret": secret}
        try:
            resp = requests.post(webhook_url, json=payload, timeout=20)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if resp.status_code == 200 and data.get("ok"):
                return True, f"Sent to {to_email}."
            return False, f"Google Apps Script error: {data.get('error') or f'HTTP {resp.status_code}'}"
        except Exception as e:
            return False, f"Couldn't send via Google Apps Script: {e}"

    if is_resend_configured():
        api_key = _get_secret("RESEND_API_KEY")
        from_email = _get_secret("RESEND_FROM_EMAIL") or "onboarding@resend.dev"
        try:
            resp = requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"from": from_email, "to": [to_email], "subject": subject, "text": body_text},
                timeout=20,
            )
            if resp.status_code in (200, 201):
                return True, f"Sent to {to_email}."
            try:
                detail = resp.json().get("message", resp.text)
            except Exception:
                detail = resp.text
            return False, f"Resend API error ({resp.status_code}): {detail}"
        except requests.exceptions.Timeout:
            return False, "Resend API request timed out."
        except requests.exceptions.ConnectionError as e:
            return False, f"Couldn't reach the Resend API — check your internet connection. ({e})"
        except Exception as e:
            return False, f"Couldn't send via Resend: {e}"

    sender_email = _get_secret("SMTP_EMAIL")
    sender_password = _get_secret("SMTP_APP_PASSWORD")
    if not sender_email or not sender_password:
        return False, ("Email sending isn't configured. Add RESEND_API_KEY to .streamlit/secrets.toml "
                        "(free at resend.com) — no SMTP ports involved.")
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain"))
        host, port = _resolve_host_port("Gmail")
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to_email], msg.as_string())
        return True, f"Sent to {to_email}."
    except Exception as e:
        return False, f"Couldn't send email: {e}"


def send_email_with_pdf(
    to_email: str, subject: str, body_text: str,
    pdf_bytes: bytes, pdf_filename: str, provider: str = "Gmail",
) -> tuple[bool, str]:
    """
    Sends an email with a PDF attachment. Returns (success, message) —
    never raises, so the caller can always show a clean result to the user
    instead of a stack trace.

    Priority: Google Apps Script (real Gmail, HTTPS) -> Resend (HTTPS) ->
    SMTP (blocked by some networks' firewalls, kept as a last resort).
    """
    if is_gas_configured():
        return _send_via_google_apps_script(to_email, subject, body_text, pdf_bytes, pdf_filename)

    if is_resend_configured():
        return _send_via_resend(to_email, subject, body_text, pdf_bytes, pdf_filename)

    import socket

    sender_email = _get_secret("SMTP_EMAIL")
    sender_password = _get_secret("SMTP_APP_PASSWORD")
    if not sender_email or not sender_password:
        return False, ("Email sending isn't configured yet. Easiest option: add RESEND_API_KEY to "
                        ".streamlit/secrets.toml (free at resend.com, works over HTTPS — no SMTP ports "
                        "involved). Or add SMTP_EMAIL and SMTP_APP_PASSWORD for traditional SMTP.")

    host, port = _resolve_host_port(provider)
    if not host:
        return False, "No SMTP host configured for a custom provider — add SMTP_HOST to secrets.toml."

    msg = _build_message(sender_email, to_email, subject, body_text, pdf_bytes, pdf_filename)

    def _try_starttls():
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to_email], msg.as_string())

    def _try_ssl():
        ssl_port = SSL_PORTS.get(provider, 465)
        with smtplib.SMTP_SSL(host, ssl_port, timeout=15) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to_email], msg.as_string())

    try:
        _try_starttls()
        return True, f"Sent to {to_email}."
    except smtplib.SMTPAuthenticationError:
        return False, ("Authentication failed — check SMTP_EMAIL/SMTP_APP_PASSWORD in secrets.toml. "
                        "Make sure you're using an app password, not your regular account password.")
    except (socket.timeout, TimeoutError, ConnectionRefusedError, OSError) as first_err:
        # Connection itself failed (not auth) — retry once over SSL before giving up,
        # since some networks block one port but allow the other.
        if provider in SSL_PORTS:
            try:
                _try_ssl()
                return True, f"Sent to {to_email}."
            except smtplib.SMTPAuthenticationError:
                return False, ("Authentication failed — check SMTP_EMAIL/SMTP_APP_PASSWORD in secrets.toml. "
                                "Make sure you're using an app password, not your regular account password.")
            except Exception:
                pass  # fall through to the timeout message below

        return False, (
            "Couldn't connect to the email server — this is almost always a network/firewall issue, "
            "not a problem with your credentials. A few things to check:\n\n"
            "- Your network, VPN, antivirus, or ISP may be blocking outbound SMTP ports (587/465) — "
            "common on some corporate or campus networks and certain routers.\n"
            "- Try a different network (e.g. mobile hotspot) to confirm this is the cause.\n"
            "- Double-check SMTP_EMAIL and the provider selected match (e.g. don't pick \"Gmail\" for a "
            "non-Gmail address).\n\n"
            f"Technical detail: {first_err}"
        )
    except Exception as e:
        return False, f"Couldn't send the email: {e}"
