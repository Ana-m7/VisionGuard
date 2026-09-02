"""Accounts, password hashing and cookie sessions for VisionGuard.

Two roles: "admin" creates users and clears history; "guard" watches feeds
and acknowledges alerts. Both see the dashboard.

Sessions are opaque random tokens kept in SQLite and handed to the browser in
an **HttpOnly cookie**, not a bearer token. That is a deliberate constraint of
this app: the camera feeds are plain `<img src="/video_feed/3">` tags, and an
image tag cannot attach an Authorization header — only a cookie rides along
automatically. Cookies also survive a page reload without keeping a token in
JavaScript where a script injection could read it.

Passwords use PBKDF2-HMAC-SHA256 from the standard library, so there is no
bcrypt/argon2 dependency to install.
"""
import hashlib
import hmac
import os
import secrets
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

ROLE_ADMIN = "admin"
ROLE_GUARD = "guard"
ROLE_LABELS = {ROLE_ADMIN: "Administrator", ROLE_GUARD: "Security Guard"}


def _connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create the account tables. Called from the app's lifespan startup,
    after the detector's init_db() has set up the event tables."""
    conn = _connect()
    cur = conn.cursor()
    if config.RESET_USERS_ON_START:
        # Lets the one-time administrator registration be demonstrated again.
        cur.execute("DROP TABLE IF EXISTS users")
        cur.execute("DROP TABLE IF EXISTS sessions")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT,
            phone TEXT,
            telegram_chat_id TEXT,
            created_at TEXT
        )
    """)
    # Migration for databases created before Telegram was a transport
    cols = [r[1] for r in cur.execute("PRAGMA table_info(users)")]
    if "telegram_chat_id" not in cols:
        cur.execute("ALTER TABLE users ADD COLUMN telegram_chat_id TEXT")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT,
            expires_at TEXT
        )
    """)
    conn.commit()
    conn.close()


# ================================================================
# PASSWORDS
# ================================================================
def hash_password(password, salt=None):
    """Return (hash_hex, salt_hex). A fresh random salt per user means two
    people with the same password get different hashes."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt),
        config.PBKDF2_ROUNDS)
    return digest.hex(), salt


def verify_password(password, expected_hash, salt):
    """Constant-time comparison — a plain == would leak how much of the hash
    matched through timing."""
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


# ================================================================
# USERS
# ================================================================
def user_count():
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return n


def create_user(username, password, role, full_name=None, phone=None,
                telegram_chat_id=None):
    """Insert a user. Raises ValueError if the username is taken."""
    pw_hash, salt = hash_password(password)
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO users
               (username, password_hash, salt, role, full_name, phone,
                telegram_chat_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (username, pw_hash, salt, role, full_name, phone, telegram_chat_id,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("That username is already taken")
    finally:
        conn.close()


def get_user_by_username(username):
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_users():
    """Every account, without password material."""
    conn = _connect()
    rows = conn.execute(
        """SELECT id, username, role, full_name, phone, telegram_chat_id,
                  created_at
           FROM users ORDER BY id""").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user(user_id):
    """Remove a user and log out every session they hold."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    removed = cur.rowcount
    conn.close()
    return removed


def reachable(role, destination_key):
    """Users of `role` who have a contact address for the active transport.

    `destination_key` is 'telegram_chat_id' or 'phone' — the notifier says
    which it needs, so switching transports changes who is reachable without
    touching the escalation logic.
    """
    if destination_key not in ("telegram_chat_id", "phone"):
        raise ValueError(f"Unknown destination field: {destination_key}")
    conn = _connect()
    rows = conn.execute(
        f"""SELECT id, username, full_name, phone, telegram_chat_id
            FROM users
            WHERE role = ? AND {destination_key} IS NOT NULL
              AND {destination_key} != ''
            ORDER BY id""", (role,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ================================================================
# SESSIONS
# ================================================================
def create_session(user_id):
    token = secrets.token_urlsafe(32)
    now = datetime.now()
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, user_id, now.strftime("%Y-%m-%d %H:%M:%S"),
         (now + timedelta(hours=config.SESSION_TTL_HOURS)).strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    return token


def user_for_token(token):
    """Resolve a session cookie to a user, or None if missing or expired."""
    if not token:
        return None
    conn = _connect()
    row = conn.execute(
        """SELECT u.id, u.username, u.role, u.full_name, u.phone, s.expires_at
           FROM sessions s JOIN users u ON u.id = s.user_id
           WHERE s.token = ?""", (token,)).fetchone()
    conn.close()
    if not row:
        return None
    if row["expires_at"] < datetime.now().strftime("%Y-%m-%d %H:%M:%S"):
        destroy_session(token)
        return None
    return {"id": row["id"], "username": row["username"], "role": row["role"],
            "full_name": row["full_name"], "phone": row["phone"]}


def destroy_session(token):
    if not token:
        return
    conn = _connect()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def purge_expired_sessions():
    conn = _connect()
    conn.execute("DELETE FROM sessions WHERE expires_at < ?",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))
    conn.commit()
    conn.close()
