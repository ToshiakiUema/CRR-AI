# scripts/check_goals_plan.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT title, learning_goals, course_plan
        FROM courses
        WHERE learning_goals IS NOT NULL OR course_plan IS NOT NULL
        LIMIT 1
        """
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        print("達成目標または授業計画が入った行が見つからない")
        return

    title, goals, plan = row
    print("科目名:", title)
    print("\n達成目標:\n", goals)
    print("\n授業計画:\n", plan)

if __name__ == "__main__":
    main()
