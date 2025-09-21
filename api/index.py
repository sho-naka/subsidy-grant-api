from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os, time
from openai import OpenAI
import asyncio
import re
import json
import logging
import traceback

app = FastAPI(title="Subsidy API")

# MVPは一旦どこからでも許可（公開時はGitHub Pagesのオリジンに絞る）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ========= 入出力スキーマ =========
class SearchRequest(BaseModel):
    prefecture: str = Field(..., description="都道府県")
    municipality: Optional[str] = Field(None, description="市区町村")
    industry: Optional[str] = Field(None, description="業種")
    keywords: Optional[str] = Field(None, description="任意キーワード")
    top_k: int = Field(10, ge=1, le=20, description="返却件数")

class GrantItem(BaseModel):
    title: str
    summary: str
    source_url: str
    grant_type: str = Field(..., description="補助金または助成金の分類")
    deadline: Optional[str] = None
    amount_max: Optional[int] = None
    rate_max: Optional[float] = None
    area: Optional[str] = None
    municipality: Optional[str] = None
    industry: Optional[str] = None
    confidence: float = 0.0
    reasons: List[str] = []

class SearchResponse(BaseModel):
    items: List[GrantItem]
    took_ms: int

# ========= OpenAI設定 =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_URL = "https://api.openai.com/v1/responses"

if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY is not set. /v1/search will fail.")

# ========= プロンプト & スキーマ =========
def build_prompt(req: SearchRequest) -> str:
        return f"""
あなたは日本の助成金・補助金の調査員です。
必ず一次情報（自治体/省庁/公的団体の公式ページ）を優先し、下記条件に合う案件を上位{req.top_k}件返してください。
金額・補助率・締切は出典に記載がある範囲のみ。憶測で作らない。要約は200字以内。

条件:
- 都道府県: {req.prefecture}
- 市区町村: {req.municipality or '指定なし'}
- 業種: {req.industry or '指定なし'}
- 追加キーワード: {req.keywords or '指定なし'}

返却は必ず厳密なJSONで、キー名は title, summary, source_url, grant_type, confidence, deadline, amount_max, rate_max, area, municipality, industry, reasons の順で出力してください。
source_url は必ず一次情報（公式のURL）を入れてください。
grant_type は「補助金」または「助成金」のいずれかを明記してください。
confidence は 0.0〜1.0 の数値で、情報の信頼度を示してください。

例（必ずこの形式に従う）:
```json
{{
    "items": [
        {{
            "title": "助成金A",
            "summary": "〜200字以内の要約",
            "source_url": "https://www.example.go.jp/...",
            "grant_type": "助成金",
            "confidence": 0.85,
            "deadline": "2025-12-31",
            "amount_max": 1000000,
            "rate_max": 0.5,
            "area": "東京都",
            "municipality": "渋谷区",
            "industry": "情報通信業",
            "reasons": ["一次情報に基づく記載あり"]
        }}
    ]
}}
```
""".strip()

def build_json_schema():
    return {
        "name": "GrantSearchResult",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["items"],
            "properties": {
                "items": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["title", "summary", "source_url", "grant_type", "confidence"],
                        "properties": {
                            "title": {"type": "string", "maxLength": 160},
                            "summary": {"type": "string", "maxLength": 220},
                            "source_url": {"type": "string"},
                            "grant_type": {"type": "string", "enum": ["補助金", "助成金"]},
                            "deadline": {"type": ["string", "null"]},
                            "amount_max": {"type": ["integer", "null"], "minimum": 0},
                            "rate_max": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                            "area": {"type": ["string", "null"]},
                            "municipality": {"type": ["string", "null"]},
                            "industry": {"type": ["string", "null"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reasons": {
                                "type": "array", "items": {"type": "string"}, "maxItems": 3
                            }
                        }
                    }
                }
            }
        }
    }

async def call_openai(prompt: str, top_k: int):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    # Use official OpenAI SDK to avoid parameter mismatches across API versions
    client = OpenAI(api_key=OPENAI_API_KEY)

    def sync_call():
        # Do not request structured schema via text to avoid Unknown parameter errors.
        return client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search"}],
            input=prompt,
            max_output_tokens=1800,
            metadata={"top_k_hint": str(top_k)},
        )

    try:
        resp = await asyncio.to_thread(sync_call)
    except Exception as e:
        tb = traceback.format_exc()
        logging.error("OpenAI call failed: %s\n%s", e, tb)
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")

    # Extract text output from response (support multiple SDK shapes)
    text = getattr(resp, "output_text", None)
    if not text:
        try:
            output = getattr(resp, "output", None) or (resp.get("output") if isinstance(resp, dict) else None)
            if output and len(output) > 0:
                first = output[0]
                content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
                if content and len(content) > 0:
                    item = content[0]
                    text = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else None)
        except Exception:
            text = None

    if not text:
        try:
            text = resp["output_text"]
        except Exception:
            text = None

    if not text:
        logging.error("No output text from OpenAI response: %s", repr(resp))
        raise HTTPException(status_code=502, detail="OpenAI returned no text output")

    # Extract JSON from model output (support fenced blocks and arrays/objects)
    def extract_json_from_text(t: str):
        # 1) try fenced code block ```json ... ``` or ``` ... ```
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.S | re.I)
        if m:
            candidate = m.group(1).strip()
            try:
                return json.loads(candidate)
            except Exception:
                # fall through to generic extractor
                pass

        # 2) find first { or [ and parse by matching braces/brackets (respecting strings)
        start = None
        for i, ch in enumerate(t):
            if ch in "{[":
                start = i
                break
        if start is None:
            return None

        stack = []
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    candidate = t[start: i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None
        return None

    parsed_json = extract_json_from_text(text)
    if parsed_json is None:
        logging.error("Failed to extract/parse JSON from model output. Snippet: %s", text[:1000])
        raise HTTPException(status_code=502, detail="Model did not return valid JSON")

    # Accept either {"items": [...]} or a bare list [...] returned by the model.
    if isinstance(parsed_json, dict) and "items" in parsed_json:
        items_list = parsed_json["items"]
    elif isinstance(parsed_json, list):
        items_list = parsed_json
    else:
        logging.error("Parsed JSON missing items or wrong shape: %s", parsed_json)
        raise HTTPException(status_code=502, detail="Unexpected OpenAI response (no items)")

    # Normalize item keys to match GrantItem fields (fallbacks for description/url etc.)
    def normalize_item(i):
        # grant_type のフォールバック（タイトルから推定）
        grant_type = i.get("grant_type")
        if not grant_type:
            title = i.get("title", "").lower()
            if "補助金" in title:
                grant_type = "補助金"
            elif "助成金" in title:
                grant_type = "助成金"
            else:
                grant_type = "補助金"  # デフォルト
        
        return {
            "title": i.get("title") or i.get("name") or "",
            "summary": i.get("summary") or i.get("description") or "",
            "source_url": i.get("source_url") or i.get("url") or i.get("link") or "",
            "grant_type": grant_type,
            "deadline": i.get("deadline") if i.get("deadline") is not None else None,
            "amount_max": i.get("amount_max") if i.get("amount_max") is not None else None,
            "rate_max": i.get("rate_max") if i.get("rate_max") is not None else None,
            "area": i.get("area") if i.get("area") is not None else None,
            "municipality": i.get("municipality") if i.get("municipality") is not None else None,
            "industry": i.get("industry") if i.get("industry") is not None else None,
            "confidence": float(i.get("confidence", 0.0)) if i.get("confidence") is not None else 0.0,
            "reasons": i.get("reasons") or [],
        }

    normalized = [normalize_item(it) for it in items_list][:top_k]

    # フィルタ: source_url が空のものは除外（全件除外される場合は元のリストを使う）
    filtered = [it for it in normalized if it.get("source_url")]
    if not filtered:
        filtered = normalized

    # confidence が 0 の場合はフォールバック値を設定（例: 0.2）
    for it in filtered:
        try:
            conf = float(it.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf == 0.0:
            it["confidence"] = 0.2

    return filtered

# ========= ルート =========
@app.get("/healthz")
async def health():
    return {"ok": True}

@app.post("/v1/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    t0 = time.time()
    items_raw = await call_openai(build_prompt(req), req.top_k)
    # Pydanticで検証しつつ整形
    items = [GrantItem(**i) for i in items_raw]
    return SearchResponse(items=items, took_ms=int((time.time() - t0) * 1000))
