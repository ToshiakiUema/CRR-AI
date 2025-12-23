# scripts/build_embeddings.py

import sqlite3
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer

DB_PATH = Path("data") / "courses.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def build_text(
    title: str | None,
    faculty: str | None,
    description: str | None,
    learning_goals: str | None,
    course_plan: str | None,
) -> str:
    parts: list[str] = []

    if title:
        parts.append(f"科目名:{title}")
    if faculty:
        parts.append(f"開講学部等:{faculty}")

    if description:
        parts.append(f"授業内容と方法:{description}")

    if learning_goals:
        parts.append(f"達成目標:{learning_goals}")

    if course_plan:
        parts.append(f"授業計画:{course_plan}")

    return "\n".join(parts)

def main() -> None:
    print("モデル読み込み中:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    conn = get_conn()
    cur = conn.cursor()

    # embeddingがまだ空のものだけ対象にする
    cur.execute(
        """
        SELECT id,title,faculty,description,learning_goals,course_plan
        FROM courses
        WHERE embedding IS NULL
        """
    )
    rows = cur.fetchall()
    print("埋める対象件数:", len(rows))

    for idx, (course_id, title, faculty, description, learning_goals, course_plan) in enumerate(rows, start=1):
        text = build_text(title, faculty, description, learning_goals, course_plan)
        if not text.strip():
            continue

        emb = model.encode(text, normalize_embeddings=True)
        emb_json = json.dumps(emb.tolist(), ensure_ascii=False)

        cur.execute(
            "UPDATE courses SET embedding=? WHERE id=?",
            (emb_json, course_id),
        )

        if idx % 10 == 0:
            conn.commit()
            print(f"{idx}件処理...")

    conn.commit()
    conn.close()
    print("埋め込み計算完了")

if __name__ == "__main__":
    main()
