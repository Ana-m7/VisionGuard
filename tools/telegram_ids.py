"""Print the Telegram chat IDs of everyone who has messaged the bot.

A Telegram bot cannot start a conversation — the person must message it
first. That first message is what makes their chat ID visible, and the chat
ID is what VisionGuard stores against a user to notify them.

Usage:

    set TELEGRAM_BOT_TOKEN=123456:ABC-your-token
    python tools/telegram_ids.py

Then paste the printed chat ID into the user's "Telegram chat ID" field in
the dashboard's User Management panel.

If nobody appears, open the bot in Telegram and press Start (or send it any
message), then run this again.
"""
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def main():
    token = config.TELEGRAM_BOT_TOKEN
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set in the environment.")
        print('  PowerShell:  $env:TELEGRAM_BOT_TOKEN = "123456:ABC-your-token"')
        return 1

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        print(f"Telegram rejected the request ({e.code}): {body}")
        if e.code == 401:
            print("  -> the bot token looks wrong. Copy it again from @BotFather.")
        return 1
    except Exception as e:
        print(f"Could not reach Telegram: {e}")
        return 1

    if not data.get("ok"):
        print("Telegram returned an error:", data.get("description"))
        return 1

    # One entry per chat; later messages overwrite earlier ones
    chats = {}
    for update in data.get("result", []):
        msg = update.get("message") or update.get("edited_message") or {}
        chat = msg.get("chat")
        if chat:
            chats[chat["id"]] = chat

    if not chats:
        print("Nobody has messaged the bot yet.")
        print("Open the bot in Telegram, press Start, then run this again.")
        print("(Telegram only keeps recent updates, so if it has been a while,")
        print(" send the bot a fresh message.)")
        return 0

    print(f"{len(chats)} chat(s) found:\n")
    print(f"{'CHAT ID':<16} NAME")
    print("-" * 46)
    for cid, chat in chats.items():
        name = " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
        name = name or chat.get("title") or chat.get("username") or "(unknown)"
        print(f"{cid:<16} {name}")
    print("\nPaste the chat ID into the user's Telegram chat ID field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
