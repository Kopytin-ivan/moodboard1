from __future__ import annotations
from typing import Dict, List
import json, re, asyncio, logging, os

from ..config import settings

log = logging.getLogger("moodboard")

SYSTEM_PROMPT = (
    "Ты — парсер дизайнерских запросов для архитектурных мудбордов. "
    "На вход — короткая фраза на русском (иногда англ.). "
    "Верни JSON РОВНО с ключами: queries_ru (array), queries_en (array). "
    "Смысл: стиль, контекст (лобби/фасад/санузел/рецепция), материалы (например, травертин), палитра/настроение. "
    "Людей/лица/логотипы/текст в запросы не добавляй — это фильтруется отдельно. "
    "По каждому языку выдай 3–6 коротких поисковых запросов."
)

def _extract_json(text: str) -> Dict[str, List[str]]:
    """
    Достаём JSON даже если модель вернула с текстом/код-блоком.
    """
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)

    def norm(arr):
        out, seen = [], set()
        for q in (arr or []):
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower()); out.append(q)
        # Возьмём не больше 6
        return out[:6]

    return {
        "queries_ru": norm(data.get("queries_ru")),
        "queries_en": norm(data.get("queries_en")),
    }

def _call_openai_sync(user_msg: str) -> Dict[str, List[str]]:
    """
    Синхронный вызов OpenAI-совместимого API через официальный SDK,
    но с base_url = https://api.aitunnel.ru/v1 (AITunnel).
    Оборачиваем его в asyncio.to_thread выше.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL.rstrip("/"),
    )

    log.info(
        f"LLM via base_url={settings.OPENAI_BASE_URL} model={settings.OPENAI_MODEL}"
    )

    resp = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Пользовательский запрос: " + user_msg},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    return _extract_json(content)

def _rule_based(prompt: str) -> Dict[str, List[str]]:
    """
    Запасной «ручной» парсер, если что-то не так с API — чтобы сервис не падал.
    """
    p = prompt.strip()
    ru = [p, f"{p} архитектура", f"{p} интерьер", f"{p} материалы", f"{p} лобби"]
    repl = {
        "сканди": "scandinavian", "минимализм": "minimalism",
        "травертин": "travertine", "лобби": "lobby", "стойка ресепшн": "reception desk",
    }
    toks = [t.strip() for t in p.split(",")]
    en = [" ".join(repl.get(t, t) for t in toks).strip(),
          "scandinavian minimalism lobby travertine",
          "stone cladding travertine lobby"]
    def norm(arr):
        out, seen = [], set()
        for q in arr:
            q=q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower()); out.append(q)
        return out[:6]
    return {"queries_ru": norm(ru), "queries_en": norm(en)}

async def build_search_queries(prompt: str, lang: str = "auto") -> Dict[str, List[str]]:
    """
    Основной вход: пробуем через AITunnel (OPENAI base_url),
    если падает — возвращаем rule-based, чтобы не ронять приложение.
    """
    user_msg = prompt

    # основной путь — AITunnel (OpenAI-совместимый)
    try:
        if (settings.LLM_PROVIDER or "").lower() == "openai" and settings.OPENAI_API_KEY:
            return await asyncio.to_thread(_call_openai_sync, user_msg)
    except Exception as e:
        log.exception(f"LLM via AITunnel failed: {e}")

    # последний шанс — rule-based
    return _rule_based(prompt)
