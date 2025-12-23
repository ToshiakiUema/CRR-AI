# scripts/check_goals_plan_embedding.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT title,
               CASE WHEN learning_goals IS NULL THEN 0 ELSE 1 END,
               CASE WHEN course_plan IS NULL THEN 0 ELSE 1 END,
               CASE WHEN embedding IS NULL THEN 0 ELSE 1 END
        FROM courses
        LIMIT 10
        """
    )

    rows = cur.fetchall()
    conn.close()

    print("title|goals|plan|embedding")
    for title, g, p, e in rows:
        print(f"{title}|{g}|{p}|{e}")

if __name__ == "__main__":
    main()
