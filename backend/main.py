# backend/main.py
from __future__ import annotations

import logging
import logging.handlers
import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .charset_fix import JsonCharsetFixMiddleware
from .config import settings

from .pipeline.prompt import build_search_queries
from .pipeline.pinterest import collect_candidates
from .pipeline.types import PinCard, MoodboardRequest, MoodboardResponse, CardOut

from .pipeline.palette import board_palette
from .pipeline.rank import total_score, deduplicate, pairwise_cosine, mmr_select
from fastapi.responses import ORJSONResponse
# Фолбэкный порядок через clip_rerank (если нет явных обёрток)
from .pipeline.embeds import clip_rerank


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


try:
    # Предпочтительный путь: явные эмбеддинги + текстовая близость
    from .pipeline.embeds import clip_image_embeddings, clip_text_similarity
    _HAVE_EMBED_WRAPPERS = True
except Exception:
    _HAVE_EMBED_WRAPPERS = False

# ---------------------------------------------------------------------
# Глобально гарантируем UTF-8 для stdout/stderr (Windows любит шалить)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

log = logging.getLogger("moodboard")

def _setup_logging_utf8():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # консоль
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s"))
        root.addHandler(sh)
    # файл
    from .config import settings
    import os
    os.makedirs(str(settings.CACHE_DIR), exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        filename=str(settings.CACHE_DIR / "server.log"),
        maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s"))
    root.addHandler(fh)

_setup_logging_utf8()

# ---------------------------------------------------------------------

app = FastAPI(
    title="moodboard-backend",
    default_response_class=ORJSONResponse
)
app.add_middleware(JsonCharsetFixMiddleware)

from starlette.middleware.base import BaseHTTPMiddleware

class ForceUTF8JSONMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        try:
            if "application/json" in (response.media_type or ""):
                # гарантируем, что клиент увидит charset
                response.headers["Content-Type"] = "application/json; charset=utf-8"
        except Exception:
            pass
        return response

app.add_middleware(ForceUTF8JSONMiddleware)


# Статика кэша изображений
images_dir: Path = settings.CACHE_DIR / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/_cache/images", StaticFiles(directory=str(images_dir), html=False), name="cache_images")

# ---------------------------------------------------------------------
# Вспомогательные утилиты

def _has_cyr(s: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", s or ""))

def _norm_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        x = [x]
    return [s.strip() for s in x if isinstance(s, str) and s.strip()]

def _preview_url_to_local_path(url: str) -> str:
    """
    Преобразует preview_url вида '/_cache/images/<file>' (или с обратными слешами)
    в абсолютный путь к файлу в кэше.
    """
    if not url:
        return ""
    s = url.replace("\\", "/")
    fname = None
    if "/_cache/images/" in s:
        fname = s.split("/_cache/images/", 1)[1]
    elif s.startswith("_cache/images/"):
        fname = s.split("_cache/images/", 1)[1]
    else:
        fname = os.path.basename(s)
    return str(images_dir / fname)

def _encoding_diag_once(q: str):
    """Диагностика кодировки: логируем строку, байты и первые codepoints (один раз на запрос)."""
    try:
        codepoints = [ord(ch) for ch in q[:40]]
        b = q.encode("utf-8", "strict")[:120]
        log.info(f"[EncodingCheck] q0='{q}' | bytes={b} | codepoints={codepoints}")
    except Exception as e:
        log.warning(f"[EncodingCheck] failed: {e}")

# ---------------------------------------------------------------------

@app.post("/api/moodboard", response_model=MoodboardResponse)
async def create_moodboard(req: MoodboardRequest):
    """
    Prompt -> queries_ru/en -> сбор кандидатов -> CLIP эмбеддинги/сходство ->
    дедупликация -> MMR-диверсификация -> палитра -> ответ.
    """
    # 1) План запросов
    plan = await build_search_queries(req.prompt, lang=req.lang)

    queries_ru: List[str] = []
    queries_en: List[str] = []

    if isinstance(plan, dict):
        queries_ru = _norm_list(plan.get("queries_ru"))
        queries_en = _norm_list(plan.get("queries_en"))
        if not queries_ru and not queries_en:
            any_q = _norm_list(plan.get("queries") or plan.get("tags") or plan.get("keywords"))
            for q in any_q:
                (queries_ru if _has_cyr(q) else queries_en).append(q)
    else:
        any_q = _norm_list(plan)
        for q in any_q:
            (queries_ru if _has_cyr(q) else queries_en).append(q)

    if not queries_ru and not queries_en:
        if _has_cyr(req.prompt):
            queries_ru = [req.prompt]
        else:
            queries_en = [req.prompt]

    queries_visible = list(dict.fromkeys([*queries_ru, *queries_en]))
    if queries_visible:
        _encoding_diag_once(queries_visible[0])

    # 2) Сбор кандидатов (внутри уже стоят фильтры лиц/текста/качества)
    need = max(req.limit * 3, req.limit + 8)

    fstats: Dict[str, int] | None = None
    res = await collect_candidates(queries_ru, queries_en, need)
    if isinstance(res, tuple):
        cards, fstats = res  # type: ignore[assignment]
    else:
        cards = res  # type: ignore[assignment]

    if not cards:
        return MoodboardResponse(prompt=req.prompt, queries=queries_visible, cards=[], filter_stats=fstats)

    # 3) Эмбеддинги и близость к тексту
    # Важно: работаем по локальным путям (из preview_url).
    local_paths = [_preview_url_to_local_path(c.preview_url) for c in cards]

    # Удалим отсутствующие файлы (на всякий случай)
    valid_pairs = [(c, p) for c, p in zip(cards, local_paths) if p and os.path.exists(p)]
    if not valid_pairs:
        return MoodboardResponse(prompt=req.prompt, queries=queries_visible, cards=[], filter_stats=fstats)
    cards, local_paths = zip(*valid_pairs)
    cards = list(cards)
    local_paths = list(local_paths)

    # Полный режим (эмбеддинги) или фолбэк (rerank)
    try:
        if _HAVE_EMBED_WRAPPERS:
            import numpy as np
            embeds = clip_image_embeddings(local_paths)             # (N,D), L2-нормированные
            sims   = clip_text_similarity(req.prompt, embeds)       # (N,)

            # 4) Грубая сортировка по total_score (эстетика подключим отдельно)
            scores = [total_score(float(s), None, None) for s in sims]
            order  = list(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True))
            cards_sorted  = [cards[i] for i in order]
            embeds_sorted = embeds[order, :]
            sims_sorted   = sims[order]

            # 5) Дедуп
            keep_idx = deduplicate([_preview_url_to_local_path(c.preview_url) for c in cards_sorted],
                                   embeds_sorted, max_phash_dist=8, emb_sim_thr=0.92)
            cards_d = [cards_sorted[i] for i in keep_idx]
            embeds_d = embeds_sorted[keep_idx, :]
            sims_d = sims_sorted[keep_idx]

            # 6) MMR
            S = pairwise_cosine(embeds_d)
            take = mmr_select(sims_d, S, k=req.limit, lam=0.7)
            final_cards  = [cards_d[i] for i in take]
            final_scores = [float(sims_d[i]) for i in take]
        else:
            # Фолбэк: только порядок через clip_rerank (без дедуп/MMR)
            idx, scores = clip_rerank(req.prompt, local_paths, top_k=len(local_paths), return_scores=True)
            ordered = [cards[i] for i in idx]
            s_arr = scores or []
            # нормируем на [0..1] для читабельности
            import numpy as np
            s = np.array(s_arr, dtype="float32")
            if s.size > 0:
                s = (s - s.min()) / (s.max() - s.min() + 1e-9)
            final_cards  = ordered[:req.limit]
            final_scores = [float(s[i]) if i < len(s) else 0.0 for i in range(len(final_cards))]
    except Exception as e:
        log.warning(f"[CLIP] embedding/similarity failed: {e}")
        final_cards = cards[:req.limit]
        final_scores = [0.0] * len(final_cards)

    # 7) Палитра по финальному набору
    final_paths = [_preview_url_to_local_path(c.preview_url) for c in final_cards]
    palette_hex = board_palette(final_paths, k=6)

    # 8) Ответ
    out_cards = [
        CardOut(
            id=c.id,
            preview_url=c.preview_url,
            source_url=c.source_url,
            score=final_scores[i] if i < len(final_scores) else 0.0
        )
        for i, c in enumerate(final_cards)
    ]

    return MoodboardResponse(
        prompt=req.prompt,
        queries=queries_visible,
        cards=out_cards,
        palette=palette_hex,
        filter_stats=fstats
    )

# ---------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    log.info(f"CLIP flags: USE_CLIP={settings.USE_CLIP}, MODEL={settings.CLIP_MODEL}, DEVICE={settings.CLIP_DEVICE}")
    log.info(f"Startup: PINTEREST_BASE_URL={settings.PINTEREST_BASE_URL}")
    log.info(f"Apify enabled? {settings.USE_APIFY} | actor={settings.APIFY_ACTOR} | token={'SET' if bool(settings.APIFY_TOKEN) else 'EMPTY'}")

@app.get("/health")
async def health():
    return {"ok": True}
