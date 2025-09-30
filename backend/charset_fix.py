# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

class JsonCharsetFixMiddleware(BaseHTTPMiddleware):
    """
    Для application/json:
    - если тело валидный UTF-8 — НЕ трогаем и возвращаем его в request._body;
    - если НЕ валидный UTF-8 — пробуем cp1251 -> перекодируем в UTF-8.
    Без эвристик по символам '?'.
    """
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        ct = (request.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            body_bytes = await request.body()
            changed = False
            try:
                # Если декодируется как UTF-8 — это ИСТИНА.
                body_bytes.decode("utf-8")  # strict
                # Важно: вернуть тело обратно, иначе downstream увидит пусто.
                request._body = body_bytes  # type: ignore[attr-defined]
            except UnicodeDecodeError:
                # Пытаемся cp1251 -> UTF-8
                try:
                    txt_cp = body_bytes.decode("cp1251")
                    json.loads(txt_cp)  # валидация JSON
                    new_bytes = txt_cp.encode("utf-8")
                    request._body = new_bytes  # type: ignore[attr-defined]
                    changed = True
                except Exception:
                    # Не удалось корректно перекодировать — оставляем как было,
                    # FastAPI вернёт 422 и это правильнее, чем ломать данные.
                    request._body = body_bytes  # type: ignore[attr-defined]

        response = await call_next(request)

        # Гарантируем charset=utf-8 в ответе JSON
        try:
            if "application/json" in (response.media_type or "") and not getattr(response, "charset", None):
                response.charset = "utf-8"
        except Exception:
            pass

        return response
