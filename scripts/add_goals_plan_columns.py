# scripts/add_goals_plan_columns.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "courses.db"

def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 達成目標
    try:
        cur.execute("ALTER TABLE courses ADD COLUMN learning_goals TEXT;")
        print("learning_goals列を追加した")
    except sqlite3.OperationalError as e:
        print("learning_goals列はスキップ:", e)

    # 授業計画
    try:
        cur.execute("ALTER TABLE courses ADD COLUMN course_plan TEXT;")
        print("course_plan列を追加した")
    except sqlite3.OperationalError as e:
        print("course_plan列はスキップ:", e)

    conn.commit()
    conn.close()
    print("列追加が完了した")

if __name__ == "__main__":
    main()
