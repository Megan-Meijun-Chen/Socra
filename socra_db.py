"""
Socra – socra_db.py
===================
SQLite session history. Stores every debate so users can revisit past verifications.

Schema:
  sessions(id, created_at, question, model_a, model_b, converged,
           final_answer, verification_summary, key_disagreement,
           total_cost, lang, result_json)
"""

import sqlite3, json, pathlib
from datetime import datetime

_DB_PATH = pathlib.Path(__file__).parent / "socra_history.db"

# ═══════════════════════════════════════════════════════════
# Init
# ═══════════════════════════════════════════════════════════

def init_db():
    """Create tables if they don't exist."""
    con = sqlite3.connect(_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at           TEXT    NOT NULL,
            question             TEXT    NOT NULL,
            model_a              TEXT    NOT NULL,
            model_b              TEXT    NOT NULL,
            converged            INTEGER NOT NULL DEFAULT 0,
            final_answer         TEXT,
            verification_summary TEXT,
            key_disagreement     TEXT,
            total_cost           REAL    DEFAULT 0.0,
            lang                 TEXT    DEFAULT 'zh',
            result_json          TEXT
        )
    """)
    con.commit()
    con.close()

# ═══════════════════════════════════════════════════════════
# Write
# ═══════════════════════════════════════════════════════════

def save_session(question: str, result: dict, lang: str = "zh") -> int:
    """
    Persist a completed debate result. Returns the new session id.
    """
    init_db()
    con = sqlite3.connect(_DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO sessions
          (created_at, question, model_a, model_b, converged,
           final_answer, verification_summary, key_disagreement,
           total_cost, lang, result_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        question[:500],
        result.get("model_a", ""),
        result.get("model_b", ""),
        1 if result.get("converged") else 0,
        result.get("final_answer", ""),
        result.get("verification_summary", ""),
        result.get("key_disagreement", "") or "",
        result.get("total_cost", 0.0),
        lang,
        json.dumps(result, ensure_ascii=False),
    ))
    con.commit()
    session_id = cur.lastrowid
    con.close()
    return session_id

# ═══════════════════════════════════════════════════════════
# Read
# ═══════════════════════════════════════════════════════════

def get_history(limit: int = 30) -> list[dict]:
    """Return recent sessions, newest first. Lightweight — no result_json."""
    init_db()
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, created_at, question, model_a, model_b,
               converged, total_cost, lang
        FROM sessions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]

def get_session(session_id: int) -> dict | None:
    """Return full session including result_json parsed back to dict."""
    init_db()
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    con.close()
    if not row:
        return None
    d = dict(row)
    if d.get("result_json"):
        try:
            d["result"] = json.loads(d["result_json"])
        except Exception:
            d["result"] = {}
    return d

def delete_session(session_id: int):
    """Delete a single session by id."""
    init_db()
    con = sqlite3.connect(_DB_PATH)
    con.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    con.commit()
    con.close()

def clear_history():
    """Delete all sessions."""
    init_db()
    con = sqlite3.connect(_DB_PATH)
    con.execute("DELETE FROM sessions")
    con.commit()
    con.close()
