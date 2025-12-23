# backend/llm_client.py
#
# LLM 切り替えモジュール（Gemini / ローカルLLM）
# - gemini: Google GenAI SDK (google-genai) を使用（公式Quickstart準拠）
# - local : llm-jp/llm-jp-3-1.8b-instruct をローカル実行
#
# provider 引数 or 環境変数 LLM_PROVIDER で切り替える

from __future__ import annotations

from typing import List, Dict, Any
import os
import traceback

from dotenv import load_dotenv
load_dotenv()  # .env を読み込む（uvicorn 起動時に自動では読まれないため）


# ============================================================
#  Gemini（google-genai / official quickstart style）
# ============================================================

try:
    from google import genai  # google-genai
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False

_genai_client = None
_genai_error: str | None = None

# Quickstart 例は gemini-2.5-flash（ただし利用可否はアカウントやプロジェクト設定で変わり得る）
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")


def _load_gemini_if_needed() -> None:
    global _genai_client, _genai_error

    if not _GENAI_AVAILABLE:
        _genai_error = (
            "google-genai がインポートできませんでした。"
            "pip install -U google-genai を確認してください。"
        )
        return

    if _genai_client is not None or _genai_error is not None:
        return

    # 新SDKは api_key を渡さなくても GEMINI_API_KEY / GOOGLE_API_KEY を拾えるが、
    # ここでは明示して読み込んでおく（デバッグしやすい）
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _genai_error = "GEMINI_API_KEY（または GOOGLE_API_KEY）が環境変数に設定されていません。"
        return

    try:
        print(f">>> [gemini] genai.Client 初期化: model={DEFAULT_GEMINI_MODEL}")
        _genai_client = genai.Client(api_key=api_key)
        print(">>> [gemini] genai.Client 初期化完了")
    except Exception as e:
        _genai_error = f"Gemini クライアント初期化に失敗しました: {e}"
        traceback.print_exc()


def _generate_with_gemini(prompt: str) -> str:
    _load_gemini_if_needed()
    if _genai_error is not None or _genai_client is None:
        raise RuntimeError(_genai_error or "Gemini が利用できません。")

    try:
        # Quickstart と同じ generateContent 相当（models.generate_content）
        resp = _genai_client.models.generate_content(
            model=DEFAULT_GEMINI_MODEL,
            contents=prompt,
        )
        text = getattr(resp, "text", None)
        if not text:
            raise RuntimeError("Gemini から text が返りませんでした。")
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API 呼び出しに失敗しました: {e}")


# ============================================================
#  Local LLM（llm-jp/llm-jp-3-1.8b-instruct）
# ============================================================

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False

LOCAL_MODEL_NAME = "llm-jp/llm-jp-3-1.8b-instruct"
_tokenizer = None
_local_model = None
_local_error: str | None = None


def _load_local_model_if_needed() -> None:
    global _tokenizer, _local_model, _local_error

    if not _TRANSFORMERS_AVAILABLE:
        _local_error = (
            "transformers/torch がインポートできませんでした。"
            "pip install torch transformers accelerate safetensors を確認してください。"
        )
        return

    if _local_model is not None or _local_error is not None:
        return

    try:
        print(f">>> [local] LLM-jp モデル読み込み開始: {LOCAL_MODEL_NAME}")

        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)

        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.bfloat16
        else:
            device_map = None
            torch_dtype = torch.float32

        _local_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_NAME,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        _local_model.eval()

        print(">>> [local] LLM-jp モデル読み込み完了")
    except Exception as e:
        _local_error = f"LLM-jp モデルのロードに失敗しました: {e}"
        traceback.print_exc()


def _generate_with_local(prompt: str) -> str:
    _load_local_model_if_needed()
    if _local_error is not None or _local_model is None or _tokenizer is None:
        raise RuntimeError(_local_error or "LLM-jp が利用できません。")

    import torch  # noqa: F401

    chat = [
        {"role": "system", "content": "あなたは大学の履修相談を行う日本語アシスタントです。"},
        {"role": "user", "content": prompt},
    ]

    input_ids = _tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(_local_model.device)

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = _local_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=220,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            temperature=0.7,
            repetition_penalty=1.05,
            pad_token_id=_tokenizer.eos_token_id,
        )[0]

    text = _tokenizer.decode(out, skip_special_tokens=True)
    return text.strip()


# ============================================================
#  Prompt
# ============================================================

def build_recommendation_prompt(user_query: str, courses: List[Dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(
        "あなたは大学の履修相談を行うアシスタントです。"
        "以下の「ユーザーの希望」と「候補授業リスト」を読み、"
        "特におすすめの授業を 3 件選び、それぞれについて日本語で分かりやすく理由を説明してください。"
    )
    lines.append("")
    lines.append("出力フォーマット:")
    lines.append("1. 授業名（講義コード）")
    lines.append("   - おすすめ理由")
    lines.append("")
    lines.append("ユーザーの希望:")
    lines.append(user_query.strip())
    lines.append("")
    lines.append("候補授業リスト:")

    for i, c in enumerate(courses, start=1):
        title = c.get("title") or "(タイトル不明)"
        code = c.get("code") or ""
        teacher = c.get("teacher") or ""
        year = c.get("year") or ""
        semester = c.get("semester") or ""
        faculty = c.get("faculty") or ""
        credits = c.get("credits") or ""
        sim = c.get("similarity")
        lines.append(
            f"{i}. {title}（{code}） 担当:{teacher} / {year} {semester} / 単位:{credits} / {faculty} / 類似度:{sim}"
        )

    lines.append("")
    lines.append("以上を踏まえて、おすすめ授業 3 件とその理由を出力してください。")
    return "\n".join(lines)


def _build_dummy(user_query: str, courses: List[Dict[str, Any]], reason: str) -> str:
    lines = [
        "※ 現在 LLM が利用できないため、ダミーを返しています。",
        f"理由: {reason}",
        "",
    ]
    for i, c in enumerate(courses[:3], start=1):
        lines.append(f"{i}. {c.get('title','(タイトル不明)')}（{c.get('code','')}）")
        lines.append(f"   - 「{user_query}」に近いと推定されました。")
        lines.append("")
    return "\n".join(lines).strip()


# ============================================================
#  Public API
# ============================================================

def generate_recommendation(
    user_query: str,
    courses: List[Dict[str, Any]],
    provider: str | None = None,
) -> str:
    """
    provider:
      - "gemini" / "local" / None（Noneなら LLM_PROVIDER を参照）
    """
    prompt = build_recommendation_prompt(user_query, courses)

    default_provider = os.getenv("LLM_PROVIDER", "local").lower()

    use = (provider or default_provider).lower()
    tried: list[str] = []
    last_err: str | None = None

    # 指定されたものを優先し、失敗したらもう片方へフォールバック
    order = ["gemini", "local"] if use == "gemini" else ["local", "gemini"]

    for p in order:
        tried.append(p)
        try:
            if p == "gemini":
                print(">>> [llm_client] Gemini で生成を試行")
                return _generate_with_gemini(prompt)
            else:
                print(">>> [llm_client] local LLM-jp で生成を試行")
                return _generate_with_local(prompt)
        except Exception as e:
            last_err = f"{p} 失敗: {e}"
            print(f"[WARN] {last_err}")
            traceback.print_exc()

    return _build_dummy(user_query, courses, reason=f"試行={tried}, 最終エラー={last_err}")
