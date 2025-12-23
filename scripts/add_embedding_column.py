# scripts/add_embedding_column.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # すでにある場合はエラーになるので、try/exceptで握る
    try:
        cur.execute("ALTER TABLE courses ADD COLUMN embedding TEXT;")
        print("embedding 列を追加した")
    except sqlite3.OperationalError as e:
        # すでに列がある場合など
        print("スキップ:", e)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
