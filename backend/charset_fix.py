# backend/charset_fix.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import json, binascii, logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

log = logging.getLogger("moodboard.charset")

def _looks_like_json_bytes(b: bytes) -> bool:
    if not b:
        return False
    # BOMы
    if b.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        return True
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 BOM
        return True
    # первый непустой символ
    ch = next((c for c in b.lstrip()[:1]), None)
    return ch in (ord("{"), ord("["))

class JsonCharsetFixMiddleware(BaseHTTPMiddleware):
    """
    Поправляет кодировку входящих JSON даже при кривом/отсутствующем Content-Type.
    Алгоритм:
      - если Content-Type содержит 'application/json' ИЛИ тело похоже на JSON,
        пробуем UTF-8 (strict), затем UTF-16 (LE/BE/auto), затем cp1251.
      - удачную декодировку пересобираем в UTF-8 для дальнейшего стандартного парсинга.
      - ответы JSON помечаем 'application/json; charset=utf-8'.
    Пишем диагностический лог и прокидываем в request.state.charset_decided/head_hex.
    """
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        ct = (request.headers.get("content-type") or "").lower()
        body_bytes = await request.body()
        should_try = ("application/json" in ct) or _looks_like_json_bytes(body_bytes)

        decided = None
        head_hex = binascii.hexlify(body_bytes[:32]).decode("ascii")

        if should_try and body_bytes:
            fixed = None

            # 1) UTF-8 strict
            try:
                body_bytes.decode("utf-8")
                fixed = body_bytes
                decided = "utf-8"
            except UnicodeDecodeError:
                pass

            # 2) UTF-16 (LE/BE/auto)
            if fixed is None:
                for enc in ("utf-16", "utf-16le", "utf-16be"):
                    try:
                        txt = body_bytes.decode(enc)
                        json.loads(txt)
                        fixed = txt.encode("utf-8")
                        decided = enc
                        break
                    except Exception:
                        continue

            # 3) cp1251
            if fixed is None:
                try:
                    txt = body_bytes.decode("cp1251")
                    json.loads(txt)
                    fixed = txt.encode("utf-8")
                    decided = "cp1251"
                except Exception:
                    pass

            # применяем, логируем
            if fixed is not None:
                request._body = fixed  # type: ignore[attr-defined]
            request.state.charset_decided = decided
            request.state.head_hex = head_hex
            log.info(f"[CharsetFix] ct='{ct}' len={len(body_bytes)} head={head_hex} decided={decided}")

        response = await call_next(request)

        # Ответы JSON — всегда с charset=utf-8
        try:
            if "application/json" in (response.media_type or ""):
                response.headers["Content-Type"] = "application/json; charset=utf-8"
        except Exception:
            pass

        return response
