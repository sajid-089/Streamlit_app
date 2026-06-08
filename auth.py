import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "auth.db")
os.makedirs(DB_DIR, exist_ok=True)


def _connect() -> sqlite3.Connection:
    """Create a database connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create users and sessions tables if they do not exist."""
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )

    conn.commit()
    conn.close()


def _hash_password(password: str, salt_hex: Optional[str] = None) -> Dict[str, str]:
    """Hash a password using PBKDF2."""
    if not salt_hex:
        salt = secrets.token_bytes(16)
    else:
        salt = bytes.fromhex(salt_hex)

    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120_000,
    )

    return {"salt": salt.hex(), "password_hash": hashed.hex()}


def _create_session(user_id: int) -> str:
    """Create a session token for a user."""
    token = secrets.token_urlsafe(32)
    created_at = datetime.utcnow()
    expires_at = created_at + timedelta(days=7)

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions (token, user_id, created_at, expires_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            token,
            user_id,
            created_at.isoformat(),
            expires_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return token


def signup(email: str, password: str, role: str = "viewer") -> Dict[str, Any]:
    """Create a new user account."""
    init_db()

    email = email.strip().lower()
    role = role.strip().lower()

    if role not in {"viewer", "analyst", "admin"}:
        role = "viewer"

    if not email or "@" not in email:
        return {"success": False, "message": "Invalid email address."}

    if len(password) < 8:
        return {"success": False, "message": "Password must be at least 8 characters."}

    conn = _connect()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE email = ?", (email,))
    if cur.fetchone():
        conn.close()
        return {"success": False, "message": "Email already exists."}

    hashed_data = _hash_password(password)

    cur.execute(
        """
        INSERT INTO users (email, password_hash, salt, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            email,
            hashed_data["password_hash"],
            hashed_data["salt"],
            role,
            datetime.utcnow().isoformat(),
        ),
    )

    user_id = cur.lastrowid
    conn.commit()
    conn.close()

    token = _create_session(user_id)

    return {
        "success": True,
        "message": "User created successfully.",
        "token": token,
        "user": {"id": user_id, "email": email, "role": role},
    }


def login(email: str, password: str) -> Dict[str, Any]:
    """Verify user credentials and create a session."""
    init_db()

    email = email.strip().lower()

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, email, password_hash, salt, role FROM users WHERE email = ?",
        (email,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return {"success": False, "message": "Invalid credentials."}

    hashed_data = _hash_password(password, row["salt"])
    if hashed_data["password_hash"] != row["password_hash"]:
        return {"success": False, "message": "Invalid credentials."}

    token = _create_session(row["id"])

    return {
        "success": True,
        "message": "Login successful.",
        "token": token,
        "user": {
            "id": row["id"],
            "email": row["email"],
            "role": row["role"],
        },
    }


def logout(token: str) -> Dict[str, Any]:
    """Remove an active session."""
    init_db()

    conn = _connect()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()

    return {"success": True, "message": "Logged out successfully."}


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """Return a user if the session token is valid and not expired."""
    init_db()

    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.email, u.role, s.expires_at
        FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token = ?
        """,
        (token,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    expires_at = datetime.fromisoformat(row["expires_at"])
    if datetime.utcnow() > expires_at:
        logout(token)
        return None

    return {
        "id": row["id"],
        "email": row["email"],
        "role": row["role"],
    }


init_db()