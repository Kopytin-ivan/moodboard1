from typing import List, Dict, Optional, Tuple, Any
import asyncio, math, logging, json, time, hashlib, re
import httpx
from pathlib import Path
from PIL import Image
from io import BytesIO
from urllib.parse import quote_plus

from ..config import settings
from .types import PinCard
from .filters import pass_all_filters

from apify_client import ApifyClientAsync

log = logging.getLogger("moodboard")

# -----------------------------------------------------------------------------#
# Константы/URLs
# -----------------------------------------------------------------------------#

APIFY_BASE_ACTOR = "apify/web-scraper"
PINTEREST_SEARCH_URL = "https://www.pinterest.com/search/pins/?q={q}&rs=typed"
BASE_URL = settings.PINTEREST_BASE_URL.rstrip("/")

HEADERS = {
    "Authorization": f"Bearer {settings.PINTEREST_API_KEY}",
    "Accept": "application/json",
}

# TLS / timeouts
_verify: object = False if settings.DEV_ONLY_SKIP_SSL_VERIFY else (settings.PINTEREST_CA_BUNDLE or True)
TIMEOUT = httpx.Timeout(
    connect=settings.TIMEOUT_SECONDS,
    read=settings.TIMEOUT_SECONDS * 2,
    write=settings.TIMEOUT_SECONDS * 2,
    pool=settings.TIMEOUT_SECONDS,
)

# Параметр-сеты для /v5/search/pins (если когда-то будет доступен)
PARAM_SETS = [
    lambda q, n: {"query": q, "page_size": min(50, n)},
    lambda q, n: {"query": q, "limit": min(50, n)},
    lambda q, n: {"q": q, "limit": min(50, n)},
    lambda q, n: {"q": q, "per_page": min(50, n)},
    lambda q, n: {"term": q, "limit": min(50, n)},
    lambda q, n: {"term": q, "page_size": min(50, n)},
]

# -----------------------------------------------------------------------------#
# Кэш кандидатов (по RU/EN запросам и limit)
# -----------------------------------------------------------------------------#

_CACHE_DIR = settings.CACHE_DIR / "pin_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_key(queries_ru: list[str], queries_en: list[str], limit: int) -> str:
    payload = {"ru": queries_ru, "en": queries_en, "limit": limit}
    s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"

def _cache_load(key: str) -> list[dict[str, Any]] | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if (time.time() - float(obj.get("ts", 0))) > settings.CACHE_TTL_SEC:
            return None
        return obj.get("cards") or None
    except Exception:
        return None

def _cache_save(key: str, cards: list[PinCard]) -> None:
    p = _cache_path(key)
    try:
        serial = [c.model_dump() for c in cards]  # pydantic BaseModel -> dict
        p.write_text(json.dumps({"ts": time.time(), "cards": serial}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.warning(f"[Cache] save failed: {e}")

# -----------------------------------------------------------------------------#
# Утилиты / парсинг ответов
# -----------------------------------------------------------------------------#

async def _http_json(client: httpx.AsyncClient, url: str, params: Dict | None = None) -> Tuple[int, Dict | None, str]:
    log.info(f"[Pinterest] -> GET {url} params={params}")
    params = {"q": query, "limit": limit}  # query — str, БЕЗ .encode() и .quote()
    r = await client.get(base, params=params, headers={"Accept": "application/json"})    
    status = r.status_code
    text_snippet = r.text[:200] if status >= 400 else ""
    try:
        data = r.json()
    except Exception:
        data = None
    log.info(f"[Pinterest] <- {status} {url}")
    return status, data, text_snippet

def _pick_preview_url(images: Dict) -> Optional[str]:
    if not isinstance(images, dict):
        return None
    for key in ("orig", "1200x", "600x", "564x", "474x", "236x"):
        if key in images and isinstance(images[key], dict) and images[key].get("url"):
            return images[key]["url"]
    return None

def _prefer_pinimg_big(u: str) -> str:
    """Если это i.pinimg.com с маленьким размером, подменяем на 736x."""
    if not isinstance(u, str) or "i.pinimg.com" not in u:
        return u
    if "/originals/" in u or "/736x/" in u:
        return u
    # Заменяем /236x/, /474x/, /564x/, /600x/, /1200x/ → /736x/
    return re.sub(r"/\d{3,4}x/", "/736x/", u)

def _parse_items(obj: Dict) -> Tuple[List[Dict], Optional[str]]:
    items = obj.get("items") or obj.get("data") or []
    bookmark = obj.get("bookmark")
    if not bookmark and isinstance(obj.get("page"), dict):
        bookmark = obj["page"].get("bookmark")
    return (items or []), bookmark

# -----------------------------------------------------------------------------#
# Загрузка превью (безопасно, с закрытием ресурса)
# -----------------------------------------------------------------------------#

async def _download_preview(client: httpx.AsyncClient, url: str, dest: Path) -> str:
    log.info(f"[Image] download start url={url} -> file={dest.name}")
    r = await client.get(url)
    r.raise_for_status()
    with Image.open(BytesIO(r.content)) as im:
        img = im.convert("RGB")
        w, h = img.size
        scale = 1024 / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, "JPEG", quality=88, optimize=True)
        final_w, final_h = img.size

    bytes_size = dest.stat().st_size if dest.exists() else 0
    web_path = f"/_cache/images/{dest.name}"
    log.info(f"[Image] download done file={dest.name} size={final_w}x{final_h} bytes={bytes_size} url={url} -> {web_path}")
    return web_path

# -----------------------------------------------------------------------------#
# Вспомогательные: статистика фильтров (воронка)
# -----------------------------------------------------------------------------#

def _init_stats() -> Dict[str, int]:
    return dict(total=0, ok=0, lowres=0, aspect=0, face=0, text=0)

def _apply_filter_stats(stats: Dict[str, int], ok: bool, reasons: Dict):
    stats["total"] += 1
    if ok:
        stats["ok"] += 1
    else:
        for k in reasons:
            stats[k] = stats.get(k, 0) + 1

def _log_stats(label: str, q: str, stats: Dict[str, int]):
    log.info(f"[FilterStats:{label}] q='{q}' total={stats['total']} ok={stats['ok']} "
             f"lowres={stats.get('lowres',0)} aspect={stats.get('aspect',0)} "
             f"face={stats.get('face',0)} text={stats.get('text',0)}")

# -----------------------------------------------------------------------------#
# APIFY (глобальный поиск)
# -----------------------------------------------------------------------------#

async def _apify_run_search_client(
    q: str,
    limit: int,
    http_client: httpx.AsyncClient | None = None,
) -> list[PinCard]:
    """Глобальный поиск через Apify/web-scraper, возвращает PinCard[]."""
    if not (settings.USE_APIFY and settings.APIFY_TOKEN):
        return []

    client = ApifyClientAsync(token=settings.APIFY_TOKEN)
    actor_id = settings.APIFY_ACTOR or APIFY_BASE_ACTOR
    log.info(f"[Apify] enabled; actor={actor_id}")

    start_url = PINTEREST_SEARCH_URL.format(q=quote_plus(q))

    # pageFunction — чистый DOM
    run_input = {
        "startUrls": [{"url": start_url}],
        "useChrome": True,
        "headless": True,
        "maxRequestsPerCrawl": 1,
        "maxRequestRetries": 0,
        "navigationTimeoutSecs": 60,
        "waitUntil": ["networkidle2"],
        "proxy": {
            "useApifyProxy": True,
            "apifyProxyCountry": (getattr(settings, "APIFY_PROXY_COUNTRY", None) or "US"),
        },
        "infiniteScroll": {
            "enabled": True,
            "maxScrollCount": 4,
            "scrollDownAndUp": False,
            "scrollDurationMillis": 1800
        },
        "pageFunction": r"""
        async function pageFunction(context) {
          const sleep = (ms) => new Promise(r => setTimeout(r, ms));
          await sleep(2000);

          const pickBestSrc = (img) => {
            const srcset = img.srcset || img.getAttribute('srcset') || '';
            if (srcset) {
              const items = srcset.split(',').map(s => s.trim());
              const prefer = items.map(s => s.split(' ')[0]).filter(Boolean);
              const originals = prefer.find(u => u.includes('/originals/'));
              if (originals) return originals;
              const big736 = prefer.find(u => /\/736x\//.test(u));
              if (big736) return big736;
              return prefer[prefer.length - 1];
            }
            return img.currentSrc || img.src || null;
          };

          const out = [];
          const seen = new Set();
          const anchors = document.querySelectorAll('a[href^="/pin/"]');

          for (const a of anchors) {
            const href = a.getAttribute('href') || '';
            const m = href.match(/\/pin\/(\d+)/);
            const id = m ? m[1] : href;
            const img = a.querySelector('img');
            if (!img) continue;

            const title = img.getAttribute('alt') || a.getAttribute('aria-label') || '';
            let src = pickBestSrc(img);
            if (!src) continue;

            if (seen.has(id)) continue;
            seen.add(id);

            const abs = href.startsWith('http') ? href : new URL(href, location.origin).href;
            out.push({ id, image: src, url: abs, title, description: title });
          }
          return out;
        }
        """
    }

    log.info(f"[Apify] call actor={actor_id} q='{q}' limit={limit}")
    actor = client.actor(actor_id)
    run = await actor.call(
        run_input=run_input,
        content_type="application/json",
        timeout_secs=max(settings.APIFY_TIMEOUT, 90),
    )

    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        log.warning("[Apify] no defaultDatasetId in run")
        return []

    ds = client.dataset(dataset_id)
    page = await ds.list_items(clean=False, limit=limit)
    raw_items = page.items or []

    # Нормализация
    norm_items: list[dict] = []
    for it in raw_items:
        if isinstance(it, dict) and "pageFunctionResult" in it:
            pfr = it.get("pageFunctionResult")
            if isinstance(pfr, list):
                norm_items.extend([x for x in pfr if isinstance(x, dict)])
            elif isinstance(pfr, dict):
                norm_items.append(pfr)
        elif isinstance(it, list):
            norm_items.extend([x for x in it if isinstance(x, dict)])
        elif isinstance(it, dict):
            norm_items.append(it)

    # HTTP клиент
    need_to_close = False
    if http_client is None:
        http_client = httpx.AsyncClient(verify=_verify, timeout=TIMEOUT, trust_env=True)
        need_to_close = True

    cards: list[PinCard] = []
    seen = set()
    stats = _init_stats()

    try:
        for it in norm_items:
            pin_id = str(it.get("id") or it.get("pin_id") or it.get("url") or hash(str(it)))
            img_url = it.get("image") or it.get("imageUrl") or it.get("image_url")
            src_url = it.get("url") or it.get("link") or ""
            if not img_url or pin_id in seen:
                continue
            seen.add(pin_id)

            dest = settings.CACHE_DIR / "images" / f"apify_{pin_id}.jpg"
            big_url = _prefer_pinimg_big(img_url)

            try:
                # Сначала пробуем 736x
                local_uri = await _download_preview(http_client, big_url, dest)
            except Exception as ex_big:
                log.warning(f"[Apify] 736x failed {big_url}: {ex_big} -> fallback to original")
                try:
                    local_uri = await _download_preview(http_client, img_url, dest)
                except Exception as ex_orig:
                    log.warning(f"[Apify] preview download failed {img_url}: {ex_orig}")
                    continue

            ok, reasons = pass_all_filters(str(dest))
            _apply_filter_stats(stats, ok, reasons)
            if not ok:
                log.info(f"[Filter] drop apify pin_id={pin_id} reasons={reasons}")
                continue

            cards.append(PinCard(
                id=pin_id,
                preview_url=local_uri,
                source_url=src_url or img_url,
                title=(it.get("title") or None),
                alt_text=(it.get("description") or None),
            ))
            if len(cards) >= limit:
                break
    finally:
        if need_to_close:
            await http_client.aclose()

    _log_stats("apify", q, stats)
    log.info(f"[Apify] parsed {len(cards)} cards for q='{q}'")
    return cards

# -----------------------------------------------------------------------------#
# Pinterest официальной API (если вдруг станет доступен)
# -----------------------------------------------------------------------------#

def _extract_pin_fields(pin: Dict) -> Optional[Dict]:
    pin_id = str(pin.get("id") or pin.get("pin_id") or pin.get("id_str") or "")
    if not pin_id:
        return None
    link = pin.get("link") or pin.get("permalink") or pin.get("url") or ""
    title = pin.get("title") or ""
    alt_text = pin.get("alt_text") or pin.get("description") or ""
    images = pin.get("images") or pin.get("media") or {}
    if isinstance(images, dict) and "images" in images:
        images = images["images"]
    preview_url = _pick_preview_url(images) or pin.get("image_large_url") or pin.get("image_url")
    if not preview_url:
        return None
    return {"id": pin_id, "preview_url": preview_url, "source_url": link or preview_url,
            "title": title, "alt_text": alt_text}

async def _search_pins_try(client: httpx.AsyncClient, q: str, limit: int) -> List[PinCard]:
    url = f"{BASE_URL}/v5/search/pins"
    cards: List[PinCard] = []
    seen = set()
    bookmark = None
    fetched = 0
    stats = _init_stats()

    while fetched < limit:
        ok_call = False
        for mk in PARAM_SETS:
            p = mk(q, limit)
            if bookmark:
                p["bookmark"] = bookmark
            status, data, snippet = await _http_json(client, url, p)
            if status == 200 and isinstance(data, dict):
                ok_call = True
                items, bookmark = _parse_items(data)
                for it in items:
                    fields = _extract_pin_fields(it)
                    if not fields:
                        continue
                    if fields["id"] in seen:
                        continue
                    seen.add(fields["id"])
                    dest = settings.CACHE_DIR / "images" / f"pin_{fields['id']}_1024.jpg"
                    try:
                        local_uri = await _download_preview(client, fields["preview_url"], dest)
                    except Exception as ex:
                        log.warning(f"[Pinterest] preview download failed {fields['preview_url']}: {ex}")
                        continue

                    ok, reasons = pass_all_filters(str(dest))
                    _apply_filter_stats(stats, ok, reasons)
                    if not ok:
                        log.info(f"[Filter] drop pin_id={fields['id']} reasons={reasons}")
                        continue

                    cards.append(PinCard(
                        id=fields["id"], preview_url=local_uri, source_url=fields["source_url"],
                        title=fields["title"] or None, alt_text=fields["alt_text"] or None
                    ))
                    fetched += 1
                    if fetched >= limit:
                        _log_stats("search_pins", q, stats)
                        return cards
                if not bookmark or fetched >= limit:
                    _log_stats("search_pins", q, stats)
                    return cards
                break
            elif status in (400, 404):
                continue
            elif status in (401, 403):
                log.warning("[Pinterest] search/pins auth/forbidden")
                _log_stats("search_pins", q, stats)
                return []
        if not ok_call:
            _log_stats("search_pins", q, stats)
            return cards

    _log_stats("search_pins", q, stats)
    return cards

async def _search_boards(client: httpx.AsyncClient, q: str, max_boards: int = 10) -> List[Dict]:
    url = f"{BASE_URL}/v5/boards/search"
    for mk in PARAM_SETS:
        params = mk(q, max_boards)
        status, data, snippet = await _http_json(client, url, params)
        if status == 200 and isinstance(data, dict):
            items, _ = _parse_items(data)
            if items:
                return items[:max_boards]
        elif status in (400, 404):
            continue
        elif status in (401, 403):
            log.warning("[Pinterest] boards/search auth/forbidden")
            return []
    return []

async def _board_pins(client: httpx.AsyncClient, board_id: str, limit: int) -> List[Dict]:
    url = f"{BASE_URL}/v5/boards/{board_id}/pins"
    pins: List[Dict] = []
    bookmark = None
    fetched = 0
    while fetched < limit:
        params = {"page_size": min(50, limit)}
        if bookmark:
            params["bookmark"] = bookmark
        status, data, snippet = await _http_json(client, url, params)
        if status == 200 and isinstance(data, dict):
            items, bookmark = _parse_items(data)
            pins.extend(items)
            fetched = len(pins)
            if not bookmark:
                break
        elif status in (400, 404):
            break
        else:
            break
    return pins[:limit]

async def _from_boards(client: httpx.AsyncClient, q: str, limit: int) -> List[PinCard]:
    cards: List[PinCard] = []
    seen = set()
    stats = _init_stats()

    boards = await _search_boards(client, q, max_boards=8)
    log.info(f"[Pinterest] boards found: {len(boards)} for query='{q}'")

    for b in boards:
        board_id = str(b.get("id") or b.get("board_id") or b.get("id_str") or "")
        if not board_id:
            continue
        raw_pins = await _board_pins(client, board_id, limit=limit)
        for it in raw_pins:
            fields = _extract_pin_fields(it)
            if not fields:
                continue
            if fields["id"] in seen:
                continue
            seen.add(fields["id"])
            dest = settings.CACHE_DIR / "images" / f"pin_{fields['id']}_1024.jpg"
            try:
                local_uri = await _download_preview(client, fields["preview_url"], dest)
            except Exception as ex:
                log.warning(f"[Pinterest] preview download failed {fields['preview_url']}: {ex}")
                continue

            ok, reasons = pass_all_filters(str(dest))
            _apply_filter_stats(stats, ok, reasons)
            if not ok:
                log.info(f"[Filter] drop board_pin id={fields['id']} reasons={reasons}")
                continue

            cards.append(PinCard(
                id=fields["id"], preview_url=local_uri, source_url=fields["source_url"],
                title=fields["title"] or None, alt_text=fields["alt_text"] or None
            ))
            if len(cards) >= limit:
                _log_stats("from_boards", q, stats)
                return cards

    _log_stats("from_boards", q, stats)
    return cards

# -----------------------------------------------------------------------------#
# Entry points
# -----------------------------------------------------------------------------#

async def _search_query_with_fallback(q: str, limit: int) -> List[PinCard]:
    async with httpx.AsyncClient(
        headers=HEADERS,
        verify=_verify,
        timeout=TIMEOUT,
        trust_env=True,
    ) as client:

        # 0) Сначала Apify (глобальный поиск)
        if settings.USE_APIFY:
            log.info(f"[Apify] enabled; actor={settings.APIFY_ACTOR}")
            ap_cards = await _apify_run_search_client(q, limit, http_client=client)
            log.info(f"[Apify] got {len(ap_cards)} cards for q='{q}'")
            if ap_cards:
                return ap_cards

        # 1) Если вдруг откроется официальный search/pins
        pins = await _search_pins_try(client, q, limit)
        if pins:
            return pins

        # 2) Fallback: boards → board pins
        pins = await _from_boards(client, q, limit)
        if pins:
            return pins

    return []

async def collect_candidates(queries_ru: List[str], queries_en: List[str], target: int) -> List[PinCard]:
    """
    Сбор кандидатов из Pinterest/APIFY.
    Порядок: RU-запросы -> EN-запросы; при PINTEREST_SERIAL=True — строго последовательно.
    Возвращает всегда List[PinCard] (даже если пустой).
    """
    try:
        need = max(settings.CANDIDATES_TARGET, target * 3)
    except Exception:
        need = target * 3

    q_all: List[str] = [q.strip() for q in (queries_ru or []) if q.strip()] + \
                       [q.strip() for q in (queries_en or []) if q.strip()]
    if not q_all:
        log.warning("[Pinterest] no queries provided")
        return []

    per_query = max(min(getattr(settings, "PIN_LIMIT_PER_QUERY", 50), need), max(1, math.ceil(need / len(q_all))))
    log.info(f"[Pinterest] (serial={getattr(settings, 'PINTEREST_SERIAL', True)}) "
             f"queries={q_all} per_query={per_query} need={need}")

    # --- КЭШ: попытка чтения перед сбором ---
    key = _cache_key(queries_ru or [], queries_en or [], need)
    cached = _cache_load(key)
    if cached:
        try:
            cards = [PinCard(**c) for c in cached]
            log.info(f"[Cache] loaded {len(cards)} cards key={key}")
            return cards[:need]
        except Exception as e:
            log.warning(f"[Cache] load failed (ignored): {e}")

    seen = set()
    all_cards: List[PinCard] = []

    # 1) Последовательный (по умолчанию): предсказуемый порядок
    if getattr(settings, "PINTEREST_SERIAL", True):
        for q in q_all:  # RU ... потом EN ...
            try:
                res = await _search_query_with_fallback(q, per_query)
            except Exception as ex:
                log.warning(f"[Pinterest] task error (serial) for '{q}': {ex}")
                res = []
            if not res:
                continue
            for c in res:
                if c and getattr(c, "id", None) and c.id not in seen:
                    seen.add(c.id)
                    all_cards.append(c)
            if len(all_cards) >= need:
                out = all_cards[:need]
                _cache_save(key, out)
                return out
        out = all_cards[:need]
        _cache_save(key, out)
        return out

    # 2) Параллельный (если явно включишь флагом): сохраняем порядок q_all
    tasks = [_search_query_with_fallback(q, per_query) for q in q_all]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, res in enumerate(results):
        if isinstance(res, Exception) or res is None:
            log.warning(f"[Pinterest] task error (parallel) for '{q_all[idx]}': {res}")
            continue
        for c in res:
            if c and getattr(c, "id", None) and c.id not in seen:
                seen.add(c.id)
                all_cards.append(c)
        if len(all_cards) >= need:
            out = all_cards[:need]
            _cache_save(key, out)
            return out

    log.info(f"[Pinterest] collected {len(all_cards)} candidates")
    out = all_cards[:need]
    _cache_save(key, out)
    return out
