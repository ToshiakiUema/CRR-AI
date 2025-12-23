# scripts/check_raw_syllabus.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# まず件数を確認
cur.execute("SELECT COUNT(*) FROM raw_syllabus;")
count = cur.fetchone()[0]
print("raw_syllabus に入っている件数:", count)

# 中身を少しだけ確認
cur.execute("SELECT id, file_name, substr(raw_text, 1, 120) FROM raw_syllabus LIMIT 3;")
rows = cur.fetchall()

for row in rows:
    print("------")
    print("id:", row[0])
    print("file:", row[1])
    print("text:", row[2].replace("\n", " "), "...")
    print("------")

conn.close()
