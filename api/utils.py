import re
import json
import logging
from typing import Any, List, Dict


def extract_json_from_text(t: str) -> Any:
    """モデルの出力テキストから JSON オブジェクトまたは配列を抽出して返します。

    手順:
    1) まず ```json ... ``` や ``` ... ``` といったコードフェンス内の JSON を優先して試します。
    2) それが見つからない場合、テキスト内で最初に現れる '{' または '[' から開始して
       括弧の対応を取ることで JSON の範囲を切り出します。文字列中のエスケープも考慮します。

    パースに成功すれば Python のデータ（dict または list）を返し、失敗すれば None を返します。
    """
    if not t:
        return None

    # 1) コードフェンス内の JSON を優先して取り出す
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.S | re.I)
    if m:
        candidate = m.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            # フェンス内でも JSON でなければフォールバックする
            pass

    # 2) 最初の { または [ から括弧の対応を見て JSON 範囲を切り出す
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
            # 文字列内ではエスケープを考慮して終了のクォートを判断
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
                    # 切り出した範囲が正しい JSON でない場合は None を返す
                    return None
    return None


def normalize_and_filter_items(items: List[Dict], top_k: int) -> List[Dict]:
    """API の GrantItem 形式に合わせて項目を正規化し、フィルタ／フォールバックを適用します。

    主な処理:
    - grant_type が無ければ title から補助金/助成金を推定（無ければ補助金をデフォルト）
    - title, summary, source_url 等のキー名の揺れを吸収して正規化
    - source_url が空のエントリは除外（全件除外になる場合は除外しない）
    - confidence が 0 または欠落している場合のフォールバック値を設定（0.2）
    - top_k による切り詰め
    """
    def normalize_item(i: Dict) -> Dict:
        # grant_type の補完（タイトルの語を見て判定）
        grant_type = i.get("grant_type")
        if not grant_type:
            title = (i.get("title") or "").lower()
            if "補助金" in title:
                grant_type = "補助金"
            elif "助成金" in title:
                grant_type = "助成金"
            else:
                grant_type = "補助金"

        normalized = {
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
        return normalized

    # 正規化して上位 top_k 件に切り詰め
    normalized = [normalize_item(it) for it in items]
    normalized = normalized[:top_k]

    # source_url が空のものを除外。ただし全件除外になる場合は元のリストを使う
    filtered = [it for it in normalized if it.get("source_url")]
    if not filtered:
        filtered = normalized

    # confidence が 0 の場合は信頼度のフォールバック値を設定
    for it in filtered:
        try:
            conf = float(it.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        if conf == 0.0:
            it["confidence"] = 0.2

    return filtered
