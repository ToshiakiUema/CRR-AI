# scripts/search_courses.py

import sqlite3
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data") / "courses.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_courses_with_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, code, title, teacher, year, semester, embedding
        FROM courses
        WHERE embedding IS NOT NULL
        """
    )
    rows = cur.fetchall()
    conn.close()

    ids = []
    infos = []
    embs = []

    for row in rows:
        course_id, code, title, teacher, year, semester, emb_json = row
        emb = np.array(json.loads(emb_json), dtype=np.float32)
        ids.append(course_id)
        infos.append((code, title, teacher, year, semester))
        embs.append(emb)

    if not embs:
        return [], [], np.zeros((0, 0), dtype=np.float32)

    embs_mat = np.vstack(embs)  # shape: (N, D)
    return ids, infos, embs_mat

def main():
    print("モデル読み込み中:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("コース埋め込み読み込み中...")
    ids, infos, embs_mat = load_courses_with_embeddings()
    if embs_mat.shape[0] == 0:
        print("embedding が入っているコースがありません。build_embeddings.py を先に実行して下さい。")
        return

    # embs_mat は normalize_embeddings=True で作っているので、行ベクトルは既に正規化済み
    # なので、クエリも normalize=True で encode すれば、内積 = コサイン類似度 になる
    while True:
        try:
            query = input("\n検索クエリ（終了するときは空Enter）：").strip()
        except EOFError:
            break

        if not query:
            print("終了します。")
            break

        q_emb = model.encode(query, normalize_embeddings=True)  # shape (D,)
        # 類似度 = 内積
        sims = embs_mat @ q_emb  # shape (N,)

        # 上位5件を取得
        top_k = 5
        idx_sorted = np.argsort(-sims)[:top_k]

        print("\n=== 検索結果 ===")
        for rank, idx in enumerate(idx_sorted, start=1):
            sim = float(sims[idx])
            code, title, teacher, year, semester = infos[idx]
            print(f"[{rank}] score={sim:.3f}")
            print(f"    コード: {code}")
            print(f"    科目名: {title}")
            print(f"    担当: {teacher}")
            print(f"    年度/学期: {year} / {semester}")
            print("")

if __name__ == "__main__":
    main()
