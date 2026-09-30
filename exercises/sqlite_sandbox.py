"""Week 2 exercise - SQLite sandbox (no AI models needed).

Creates data/sandbox.sqlite3 with a small ``documents`` table and walks through
CREATE / INSERT / SELECT / UPDATE / DELETE, including storing a vector as a
BLOB and reading it back - exactly what the real pipeline does.

    python exercises/sqlite_sandbox.py
Afterwards you can explore it with the sqlite3 CLI:
    sqlite3 data/sandbox.sqlite3 "SELECT id, title FROM documents;"
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

DB = Path(__file__).resolve().parent.parent / "data" / "sandbox.sqlite3"


def show(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> None:
    print(f"\nsqlite> {sql}  {params if params else ''}")
    for row in conn.execute(sql, params):
        print("   ", tuple(row))


def main() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)  # start fresh every run
    conn = sqlite3.connect(DB)

    conn.execute(
        """CREATE TABLE documents (
               id        INTEGER PRIMARY KEY,
               title     TEXT NOT NULL,
               content   TEXT NOT NULL,
               embedding BLOB            -- float32 vector stored as raw bytes
           )"""
    )

    rows = [
        ("Power supply", "The board is powered via USB or an external 5 V supply."),
        ("LEDs", "Four user LEDs are connected to GPIO port D."),
        ("Push-button", "The blue user button is connected to PA0."),
    ]
    rng = np.random.default_rng(0)
    with conn:  # commit automatically
        for title, content in rows:
            vec = rng.standard_normal(4).astype(np.float32)  # pretend embedding
            conn.execute(
                "INSERT INTO documents(title, content, embedding) VALUES (?, ?, ?)",
                (title, content, vec.tobytes()),
            )

    show(conn, "SELECT id, title FROM documents")
    show(conn, "SELECT title, content FROM documents WHERE id = ?", (2,))
    show(conn, "SELECT title FROM documents WHERE content LIKE ?", ("%GPIO%",))
    show(conn, "SELECT COUNT(*) FROM documents")

    blob = conn.execute("SELECT embedding FROM documents WHERE id = 1").fetchone()[0]
    vec = np.frombuffer(blob, dtype=np.float32)
    print(f"\nVector read back from BLOB ({len(blob)} bytes): {[round(float(x), 3) for x in vec]}")

    with conn:
        conn.execute("UPDATE documents SET title = ? WHERE id = ?", ("User LEDs", 2))
        conn.execute("DELETE FROM documents WHERE id = ?", (3,))
    show(conn, "SELECT id, title FROM documents")

    conn.close()
    print(f"\nDatabase file: {DB}")


if __name__ == "__main__":
    main()
