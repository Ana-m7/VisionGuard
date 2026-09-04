"""Escalation ladder for unacknowledged alerts.

An alert that nobody acknowledges climbs on its own:

    alert fires
        ── ESCALATE_GUARD_SECONDS unacknowledged ──► notify security guards
        ── ESCALATE_ADMIN_SECONDS unacknowledged ──► notify administrators

Each alert row carries its own `escalation_level` (0 → 1 → 2) and
`escalated_at`, so every alert climbs independently: two alerts raised
seconds apart escalate on their own clocks, and acknowledging one never
affects another. Levels only ever move forward, and the scanner claims a row
by writing the new level before sending, so a message is never sent twice
even if a sweep runs long.

Acknowledging an alert at any point stops its climb — the queries require
`acknowledged_at IS NULL`.
"""
import asyncio
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backend import auth
from backend.notify import notifier

# alert type → (table, alert flag column, human label, only_while_active)
#
# The last field encodes the difference between a *condition* and an
# *incident*, which decides whether a situation that has already ended is
# still worth someone's phone buzzing:
#
#   Crowd gathering is a condition. If the crowd dispersed before anyone was
#   notified, there is nothing left to attend to — notifying then is the
#   "why is my phone going off, everything looks fine" case.
#
#   A fall is an incident. It is over in about a second by its nature: the
#   person is on the ground, or they got up. Suppressing it because it "ended"
#   would suppress every fall the system will ever see, which is the one thing
#   this system exists to report.
def _tables():
    """Rebuilt each sweep so the ESCALATE_ENDED_CROWDS flag can be flipped
    without a restart. `only_while_active` = suppress if the situation has
    since cleared."""
    return {
        "gathering": ("crowd_events", "is_alert", "CROWD GATHERING",
                      not config.ESCALATE_ENDED_CROWDS),
        "fall": ("fall_events", "is_fall", "FALL DETECTED", False),
    }

LEVEL_GUARD, LEVEL_ADMIN = 1, 2

# (table, cam_id, level) → when we last actually notified about it. In memory
# rather than in the DB: it is a courtesy to the recipient's phone, not part
# of the incident record, and starting a fresh run with a clean slate is the
# behaviour you want.
_last_notified = {}


def _should_notify(table, cam_id, level, now=None):
    """Group repeats: one notification per camera+type+rung per window.

    The alert is still escalated and still recorded either way — this only
    decides whether a phone buzzes, so a repeating situation cannot turn into
    a stream of identical messages.
    """
    now = now or time.time()
    key = (table, cam_id, level)
    if now - _last_notified.get(key, -1e9) < config.NOTIFY_GROUP_SECONDS:
        return False
    _last_notified[key] = now
    return True


def _connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _cutoff(seconds):
    return (datetime.now() - timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")


def _due(conn, table, flag, level, seconds, only_while_active):
    """Alerts that have sat unacknowledged past this rung's delay.

    Bounded at both ends: older than the rung's delay, but not older than
    ESCALATE_MAX_AGE_SECONDS. The upper bound matters because a backlog of
    stale unacknowledged alerts would otherwise all come due at once and
    fire a burst of notifications.
    """
    active_clause = (f"""AND NOT EXISTS (
                  SELECT 1 FROM {table} AS later
                  WHERE later.cam_id IS due.cam_id
                    AND later.id > due.id
                    AND later.{flag} = 0
              )""" if only_while_active else "")
    return conn.execute(
        f"""SELECT id, cam_id, timestamp FROM {table} AS due
            WHERE {flag} = 1
              AND acknowledged_at IS NULL
              AND COALESCE(escalation_level, 0) = ?
              AND timestamp <= ?
              AND timestamp >= ?
              -- Conditions only: skip if the detector has since written a
              -- clear row for this camera, meaning the crowd dispersed before
              -- anyone was notified. Incidents (falls) skip this check — they
              -- end in about a second by nature, and suppressing them for
              -- having ended would suppress every fall the system ever sees.
              {active_clause}
              -- Acknowledgement suppression: stay quiet about a camera whose
              -- operator has recently acknowledged this kind of alert. A
              -- continuing situation keeps producing new alert rows, and
              -- without this the operator is notified again seconds after
              -- confirming they are on it.
              AND NOT EXISTS (
                  SELECT 1 FROM {table} AS ack
                  WHERE ack.cam_id IS due.cam_id
                    AND ack.acknowledged_at IS NOT NULL
                    AND ack.acknowledged_at >= ?
              )
            ORDER BY id""",
        (level - 1, _cutoff(seconds),
         _cutoff(config.ESCALATE_MAX_AGE_SECONDS),
         _cutoff(config.ACK_SUPPRESS_SECONDS))).fetchall()


def _claim(conn, table, row_id, level):
    """Move the row to `level`, but only from the level below. The WHERE
    clause makes this atomic: if two sweeps overlap, exactly one wins and
    the other sends nothing."""
    cur = conn.cursor()
    cur.execute(
        f"""UPDATE {table} SET escalation_level = ?, escalated_at = ?
            WHERE id = ? AND COALESCE(escalation_level, 0) = ?""",
        (level, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), row_id, level - 1))
    conn.commit()
    return cur.rowcount > 0


def _message(label, cam_id, when, level, waited):
    cam = f"CAM-{cam_id:02d} {config.CAMERA_LABELS.get(cam_id, '')}".strip()
    head = ("VisionGuard alert" if level == LEVEL_GUARD
            else "VisionGuard ESCALATION")
    tail = ("No response from the security guard."
            if level == LEVEL_ADMIN else "Please respond.")
    return (f"{head}\n{label} at {cam}\nTime: {when}\n"
            f"Unacknowledged for {waited}s. {tail}")


def _recipients(level):
    """Who to notify at this rung, addressed for whichever transport is
    active — Telegram wants a chat ID, WhatsApp wants a phone number."""
    role = auth.ROLE_GUARD if level == LEVEL_GUARD else auth.ROLE_ADMIN
    return auth.reachable(role, notifier.destination_key)


def _escalate_once(push_event):
    """One sweep over both tables and both rungs. Returns how many alerts
    were escalated. Runs in a worker thread — it does blocking DB and HTTP."""
    escalated = 0
    conn = _connect()
    try:
        for level, seconds in ((LEVEL_GUARD, config.ESCALATE_GUARD_SECONDS),
                               (LEVEL_ADMIN, config.ESCALATE_ADMIN_SECONDS)):
            people = _recipients(level)
            for alert_type, (table, flag, label, only_active) in _tables().items():
                for row in _due(conn, table, flag, level, seconds, only_active):
                    if not _claim(conn, table, row["id"], level):
                        continue                      # another sweep took it
                    escalated += 1
                    sent, grouped = [], False
                    if _should_notify(table, row["cam_id"], level):
                        text = _message(label, row["cam_id"], row["timestamp"],
                                        level, seconds)
                        for person in people:
                            if notifier.send(person[notifier.destination_key], text):
                                sent.append(person["full_name"] or person["username"])
                        if not people:
                            who = 'guard' if level == LEVEL_GUARD else 'administrator'
                            print(f"  no {who} has a {notifier.destination_key} on file "
                                  f"— nothing to notify")
                    else:
                        grouped = True
                    push_event({
                        "type": "escalation",
                        "alert_type": alert_type,
                        "id": row["id"],
                        "cam_id": row["cam_id"],
                        "level": level,
                        "waited": seconds,
                        "notified": sent,
                        "grouped": grouped,
                        "recipients": len(people),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
    finally:
        conn.close()
    return escalated


async def escalation_loop(push_event):
    """Background sweep. Sleeps between passes so it costs nothing when the
    system is quiet, and never lets an exception kill the loop."""
    print(f"Escalation ladder active: guard at {config.ESCALATE_GUARD_SECONDS}s, "
          f"administrator at {config.ESCALATE_ADMIN_SECONDS}s "
          f"(sweeping every {config.ESCALATION_SCAN_SECONDS}s)", flush=True)
    sweeps = 0
    while True:
        try:
            n = await asyncio.to_thread(_escalate_once, push_event)
            sweeps += 1
            # A quiet heartbeat every ~2 min proves the loop is alive even
            # when nothing is escalating.
            if n or sweeps % max(1, int(120 / config.ESCALATION_SCAN_SECONDS)) == 0:
                print(f"[escalation] sweep #{sweeps}: {n} alert(s) escalated",
                      flush=True)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            import traceback
            print(f"[escalation] sweep FAILED: {e}", flush=True)
            traceback.print_exc()
        await asyncio.sleep(config.ESCALATION_SCAN_SECONDS)
