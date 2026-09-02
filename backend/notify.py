"""Outbound notifications for escalated alerts.

Transport is Twilio's WhatsApp sandbox, reached over plain HTTP rather than
the Twilio SDK — one stdlib POST, no extra dependency to install on a demo
machine.

WhatsApp rather than SMS for a regulatory reason, not a technical one:
sending A2P SMS to Indian numbers requires TRAI DLT registration (a paid
registry with 2-4 week sender-ID provisioning and a registered business
entity behind it). The WhatsApp sandbox delivers to a real phone today, for
free, with no registration. Swapping in SMS later is a change to one class.

Every send is best-effort. If credentials are missing, the network is down,
or Twilio rejects the message, the notifier logs what it would have sent and
reports failure — it never raises into the escalation loop, so a bad key or
a dropped Wi-Fi connection cannot take the system down mid-incident.
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
TIMEOUT_SECONDS = 10


def _log(prefix, to, message):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] NOTIFY {prefix} -> {to}: {message}")


class LogNotifier:
    """Fallback transport: prints the message instead of sending it. Used
    whenever Twilio is not configured, so the escalation ladder is fully
    demonstrable with no account at all."""

    name = "log"
    destination_key = "phone"
    configured = True

    def send(self, to_phone, message):
        _log("(log only)", to_phone, message)
        return True


class TelegramNotifier:
    """Telegram Bot API — the transport the demo actually runs on.

    Chosen over WhatsApp for a practical reason: Twilio's free tier rejects
    custom message bodies (error 21654 — trial accounts may only send
    Twilio's own templates such as appointment reminders), and the Content
    API needed to author a template requires a paid plan. Telegram delivers
    arbitrary text to a real phone, free, with no templates, approval flow or
    per-message quota.

    Recipients are addressed by chat ID rather than phone number: a person
    messages the bot once, and `tools/telegram_ids.py` reads the chat ID back
    out of getUpdates. A bot cannot message someone who has never contacted
    it, which is Telegram's anti-spam rule.
    """

    name = "telegram"
    destination_key = "telegram_chat_id"
    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self.token = config.TELEGRAM_BOT_TOKEN
        self.configured = bool(self.token)
        self._fallback = LogNotifier()

    def send(self, to_chat_id, message):
        if not self.configured:
            return self._fallback.send(to_chat_id, message)
        body = urllib.parse.urlencode({
            "chat_id": str(to_chat_id),
            "text": message,
            "disable_web_page_preview": "true",
        }).encode()
        req = urllib.request.Request(
            self.API.format(token=self.token), data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
                payload = json.loads(r.read().decode())
            if payload.get("ok"):
                _log("(telegram sent)", to_chat_id, message)
                return True
            print(f"NOTIFY FAILED for {to_chat_id}: {payload.get('description')}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:200]
            print(f"NOTIFY FAILED ({e.code}) for {to_chat_id}: {detail}")
            if "chat not found" in detail:
                print("  -> that chat ID has never messaged the bot. Open the bot "
                      "in Telegram and press Start, then re-read the chat ID.")
        except Exception as e:
            print(f"NOTIFY FAILED for {to_chat_id}: {e}")
        self._fallback.send(to_chat_id, message)
        return False


class TwilioWhatsAppNotifier:
    """Twilio WhatsApp sandbox.

    The recipient must first join the sandbox by sending `join <code>` to the
    sandbox number; that opens a 24-hour window in which free-form messages
    are delivered at no charge. Messaging a number that has not joined fails
    with Twilio error 63015 — which is logged, not raised.
    """

    name = "twilio-whatsapp"
    destination_key = "phone"

    def __init__(self):
        self.sid = config.TWILIO_ACCOUNT_SID
        self.token = config.TWILIO_AUTH_TOKEN
        self.sender = config.TWILIO_WHATSAPP_FROM
        self.configured = bool(self.sid and self.token and self.sender)
        self._fallback = LogNotifier()

    @staticmethod
    def _wa(number):
        """Twilio addresses WhatsApp endpoints as 'whatsapp:+<E.164>'."""
        number = number.strip()
        return number if number.startswith("whatsapp:") else f"whatsapp:{number}"

    def send(self, to_phone, message):
        if not self.configured:
            return self._fallback.send(to_phone, message)
        body = urllib.parse.urlencode({
            "From": self._wa(self.sender),
            "To": self._wa(to_phone),
            "Body": message,
        }).encode()
        req = urllib.request.Request(
            TWILIO_API.format(sid=self.sid), data=body, method="POST")
        cred = base64.b64encode(f"{self.sid}:{self.token}".encode()).decode()
        req.add_header("Authorization", f"Basic {cred}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
                payload = json.loads(r.read().decode())
            _log(f"(whatsapp {payload.get('status', 'sent')})", to_phone, message)
            return True
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:200]
            print(f"NOTIFY FAILED ({e.code}) for {to_phone}: {detail}")
            if b"63015" in detail.encode():
                print("  -> that number has not joined the WhatsApp sandbox; "
                      "send 'join <code>' to the sandbox number first.")
        except Exception as e:
            print(f"NOTIFY FAILED for {to_phone}: {e}")
        # Delivery failed — still record the message so the incident trail
        # and the demo both survive a network problem.
        self._fallback.send(to_phone, message)
        return False


class RateLimited:
    """Caps real sends per rolling hour, logging anything over the ceiling.

    Twilio trial accounts have a small fixed message allowance, and looping
    demo footage can raise an alert every few seconds. Left unbounded, an
    afternoon of testing with nobody acknowledging would quietly exhaust the
    quota and the notifications would be dead by presentation time.
    """

    def __init__(self, inner, max_per_hour):
        self._inner = inner
        self._max = max_per_hour
        self._sent = deque()          # timestamps of real sends
        self._lock = threading.Lock()
        self._fallback = LogNotifier()
        self.name = f"{inner.name} (max {max_per_hour}/h)"
        self.configured = inner.configured
        self.destination_key = inner.destination_key

    def _take_slot(self):
        with self._lock:
            cutoff = time.time() - 3600
            while self._sent and self._sent[0] < cutoff:
                self._sent.popleft()
            if len(self._sent) >= self._max:
                return False
            self._sent.append(time.time())
            return True

    def remaining(self):
        with self._lock:
            cutoff = time.time() - 3600
            recent = sum(1 for t in self._sent if t >= cutoff)
        return max(0, self._max - recent)

    def send(self, to_phone, message):
        if not self._take_slot():
            print(f"NOTIFY THROTTLED ({self._max}/hour reached) — logging instead")
            return self._fallback.send(to_phone, message)
        return self._inner.send(to_phone, message)


def build_notifier():
    """Pick a transport at startup.

    Telegram first: it is the only free option that delivers our own alert
    wording. WhatsApp is kept wired behind it, and NOTIFY_TRANSPORT can pin
    either one explicitly. With neither configured everything is logged, so
    the escalation ladder is still fully demonstrable with no accounts.
    """
    want = (config.NOTIFY_TRANSPORT or "auto").lower()
    telegram, twilio = TelegramNotifier(), TwilioWhatsAppNotifier()

    if want in ("telegram", "auto") and telegram.configured:
        print(f"Notifications: Telegram "
              f"(capped at {config.NOTIFY_MAX_PER_HOUR}/hour)")
        return RateLimited(telegram, config.NOTIFY_MAX_PER_HOUR)
    if want in ("whatsapp", "auto") and twilio.configured:
        print(f"Notifications: Twilio WhatsApp, from {twilio.sender} "
              f"(capped at {config.NOTIFY_MAX_PER_HOUR}/hour)")
        return RateLimited(twilio, config.NOTIFY_MAX_PER_HOUR)
    if want == "log":
        print("Notifications: log only (pinned by NOTIFY_TRANSPORT)")
    else:
        print("Notifications: log only (set TELEGRAM_BOT_TOKEN in the "
              "environment to send real Telegram messages)")
    return LogNotifier()


notifier = build_notifier()
