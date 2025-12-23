# scripts/check_description_full.py

import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
    SELECT title, description
    FROM courses
    WHERE description IS NOT NULL
    LIMIT 1
""")

row = cur.fetchone()
conn.close()

if row:
    title, description = row
    print("【科目名】")
    print(title)
    print("\n【description】")
    print(description)
else:
    print("description が入っているデータがありません")
