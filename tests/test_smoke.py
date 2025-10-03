# tests/test_smoke.py
import os
import httpx
import pytest

BASE = os.getenv("BASE", "http://127.0.0.1:8000")

@pytest.mark.asyncio
async def test_health():
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/health")
        assert r.status_code == 200
        j = r.json()
        assert j.get("ok") is True

@pytest.mark.asyncio
async def test_engines():
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/api/debug/engines")
        assert r.status_code == 200
        j = r.json()
        # CLIP: либо есть, либо ошибка — но ручка обязана отработать
        assert "clip" in j or "clip_error" in j
        assert "ocr_backends" in j or "ocr_error" in j
        assert "apify" in j

@pytest.mark.asyncio
async def test_pins_and_filters_minimal():
    # этот эндпоинт у тебя уже есть как debug/pins; если его нет — пропусти тест
    payload = {
        "queries_ru": ["сканди минимализм лобби травертин"],
        "queries_en": ["scandinavian minimalism lobby travertine"],
        "target": 12
    }
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(f"{BASE}/api/debug/pins", json=payload)
        if r.status_code == 404:
            pytest.skip("Нет /api/debug/pins — пропускаем тест")
        assert r.status_code == 200
        j = r.json()
        assert "total" in j
        if j["total"] > 0:
            sample = j["sample"][0]
            assert sample["preview_url"].startswith("/_cache/images/")
            # Проверим фильтр на одном файле
            r2 = await c.post(f"{BASE}/api/debug/filter_one", json={"path": sample["preview_url"]})
            assert r2.status_code == 200
            jj = r2.json()
            assert "pass_all" in jj
