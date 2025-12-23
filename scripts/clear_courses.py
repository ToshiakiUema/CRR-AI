# scripts/clear_courses.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # courses テーブルがなければ何もしないで終わる
    cur.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id      INTEGER,
            code        TEXT,
            title       TEXT,
            teacher     TEXT,
            year        INTEGER,
            semester    TEXT,
            description TEXT
        );
    """)

    cur.execute("DELETE FROM courses;")
    conn.commit()
    conn.close()
    print("courses テーブルのデータをすべて削除した")

if __name__ == "__main__":
    main()
