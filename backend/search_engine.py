# backend/search_engine.py

import sqlite3
from pathlib import Path
import json
from typing import List, Tuple, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = Path("data") / "courses.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SearchEngine:
    def __init__(self) -> None:
        print(">>> SearchEngine 起動: モデル読み込み中:", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME)
        print(">>> モデル読み込み完了")
        print(">>> DB から埋め込み読み込み中...")
        self.ids, self.infos, self.embs = self._load_courses_with_embeddings()
        print(f">>> {len(self.ids)} 件のコースをロード")

    def _load_courses_with_embeddings(
        self,
    ) -> Tuple[List[int], List[Dict[str, Any]], np.ndarray]:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                id,
                code,
                subject_number,
                title,
                teacher,
                year,
                semester,
                faculty,
                classroom,
                day_period,
                credits,
                embedding
            FROM courses
            WHERE embedding IS NOT NULL
            """
        )
        rows = cur.fetchall()
        conn.close()

        ids: List[int] = []
        infos: List[Dict[str, Any]] = []
        embs: List[np.ndarray] = []

        for row in rows:
            (
                course_id,
                code,
                subject_number,
                title,
                teacher,
                year,
                semester,
                faculty,
                classroom,
                day_period,
                credits,
                emb_json,
            ) = row

            emb = np.array(json.loads(emb_json), dtype=np.float32)

            ids.append(course_id)
            infos.append(
                {
                    "id": course_id,
                    "code": code,
                    "subject_number": subject_number,
                    "title": title,
                    "teacher": teacher,
                    "year": year,
                    "semester": semester,
                    "faculty": faculty,
                    "classroom": classroom,
                    "day_period": day_period,
                    "credits": credits,
                }
            )
            embs.append(emb)

        if not embs:
            return [], [], np.zeros((0, 0), dtype=np.float32)

        embs_mat = np.vstack(embs)  # shape: (N, D)
        return ids, infos, embs_mat

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embs.shape[0] == 0:
            return []

        # クエリをベクトル化（正規化付き）
        q_emb = self.model.encode(query, normalize_embeddings=True)  # (D,)
        sims = self.embs @ q_emb  # 内積 = コサイン類似度 (N,)

        top_k = max(1, min(top_k, len(self.ids)))
        idx_sorted = np.argsort(-sims)[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, idx in enumerate(idx_sorted, start=1):
            score = float(sims[idx])
            info = self.infos[idx]
            result = {
                "rank": rank,
                "score": score,
                **info,
            }
            results.append(result)
        return results
