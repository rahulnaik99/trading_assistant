"""Auth — user store (SQLite), password hashing, JWT tokens."""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from backend.config import settings

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/users.db")
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

ALGORITHM = "HS256"


# ── DB bootstrap ──────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                uid       TEXT PRIMARY KEY,
                username  TEXT UNIQUE NOT NULL,
                hashed_pw TEXT NOT NULL,
                created   TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                uid         TEXT NOT NULL,
                created     TEXT NOT NULL,
                FOREIGN KEY(uid) REFERENCES users(uid)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                tokens      INTEGER DEFAULT 0,
                ts          TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            )
        """)
        logger.info("auth.init_db: tables ready at %s", _DB_PATH)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(username: str, password: str) -> str:
    import uuid
    uid = str(uuid.uuid4())
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO users(uid,username,hashed_pw,created) VALUES(?,?,?,?)",
                (uid, username, hashed, datetime.now(timezone.utc).isoformat()),
            )
        logger.info("auth: user created  username=%s", username)
        return uid
    except sqlite3.IntegrityError:
        raise ValueError(f"Username {username!r} already exists")


def verify_user(username: str, password: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        return None
    if not bcrypt.checkpw(password.encode(), row["hashed_pw"].encode()):
        return None
    return dict(row)


# ── JWT tokens ────────────────────────────────────────────────────────────────

def create_access_token(uid: str, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(
        {"sub": uid, "username": username, "exp": expire},
        settings.SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


# ── Sessions + message history ────────────────────────────────────────────────

def create_session(uid: str) -> str:
    import uuid
    sid = str(uuid.uuid4())
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions(session_id,uid,created) VALUES(?,?,?)",
            (sid, uid, datetime.now(timezone.utc).isoformat()),
        )
    return sid


def list_sessions(uid: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT session_id, created FROM sessions WHERE uid=? ORDER BY created DESC",
            (uid,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_message(session_id: str, role: str, content: str, tokens: int = 0) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO messages(session_id,role,content,tokens,ts) VALUES(?,?,?,?,?)",
            (session_id, role, content, tokens, datetime.now(timezone.utc).isoformat()),
        )


def load_messages(session_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role,content,tokens,ts FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def session_total_tokens(session_id: str) -> int:
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(tokens),0) AS t FROM messages WHERE session_id=?",
            (session_id,),
        ).fetchone()
    return int(row["t"])
