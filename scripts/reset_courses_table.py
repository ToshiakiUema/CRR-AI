# scripts/reset_courses_table.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # もし courses テーブルがあれば削除
    cur.execute("DROP TABLE IF EXISTS courses;")
    conn.commit()
    conn.close()
    print("courses テーブルを削除しました")

if __name__ == "__main__":
    main()
