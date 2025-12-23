# backend/main.py
#
# FastAPI バックエンド本体
# - /search     : コサイン類似度で授業検索（SearchEngine 使用）
# - /recommend  : Gemini / ローカル LLM の切り替え推薦生成
# - /health     : 動作確認

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal, List, Dict, Any

from .search_engine import SearchEngine
from .llm_client import generate_recommendation


# ==============================
# FastAPI 初期化
# ==============================

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# SearchEngine を 1 回だけ起動
# ==============================

search_engine = SearchEngine()


def search_courses(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    search_engine.search() の結果を、フロントに返しやすい形に整える。
    search_engine.search は "score" を返すため、フロント側で使う "similarity"
    キーにリネームして返す。
    """
    raw = search_engine.search(query, top_k=top_k)

    results: List[Dict[str, Any]] = []
    for r in raw:
        results.append(
            {
                "id": r.get("id"),
                "code": r.get("code"),
                "title": r.get("title"),
                "teacher": r.get("teacher"),
                "year": r.get("year"),
                "semester": r.get("semester"),
                "faculty": r.get("faculty"),
                "credits": r.get("credits"),
                "similarity": float(r.get("score", 0.0)),
                # おまけ情報（使うかは自由）
                "rank": r.get("rank"),
                "subject_number": r.get("subject_number"),
                "classroom": r.get("classroom"),
                "day_period": r.get("day_period"),
            }
        )

    return results


# ==============================
# Request / Response モデル
# ==============================

class RecommendRequest(BaseModel):
    query: str
    top_k: int = 10
    provider: Optional[Literal["local", "gemini"]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


# ==============================
# /health
# ==============================

@app.get("/health")
def health():
    return {"status": "ok"}


# ==============================
# /search（コサイン類似度検索）
# ==============================

@app.post("/search")
def search(req: SearchRequest):
    courses = search_courses(req.query, req.top_k)
    return {"courses": courses}


# ==============================
# /recommend（LLM による推薦文生成）
# ==============================

@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    provider に応じて LLM を選択：
      - "local"  : LLM-jp（ローカル）
      - "gemini" : Gemini API
      - None     : .env の LLM_PROVIDER に従う
    """

    # ① コサイン類似度検索
    courses = search_courses(req.query, req.top_k)

    # ② LLM による推薦文生成
    summary = generate_recommendation(
        user_query=req.query,
        courses=courses,
        provider=req.provider,
    )

    return {
        "summary": summary,
        "courses": courses,
    }
