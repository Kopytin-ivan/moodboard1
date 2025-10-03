from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

# импортируем всё из твоего filters.py
from .filters import (
    has_text, has_face, is_lowres, bad_aspect, pass_all_filters,
)
from . import filters as _f  # чтобы достать внутреннее: _probe_ocr_backends_once, _haar

# --- генераторы тестовых картинок ---
def _mk_dir() -> Path:
    d = Path(".selftest_filters")
    d.mkdir(exist_ok=True)
    return d

def _img_text(path: Path) -> Path:
    # белый фон + жирный чёрный текст
    h, w = 600, 1200
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, "HELLO LOBBY 123", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(img, "TRAVERTINE", (40, 380), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (10, 10, 10), 8, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return path

def _img_lowres(path: Path, short_side: int = 128) -> Path:
    # квадратик меньше MIN_SHORT_SIDE
    img = np.full((short_side - 10, short_side - 10, 3), 200, np.uint8)
    cv2.imwrite(str(path), img)
    return path

def _img_bad_aspect(path: Path) -> Path:
    # экстремально вытянутое изображение
    img = np.full((120, 3600, 3), 230, np.uint8)
    cv2.imwrite(str(path), img)
    return path

def _img_clean_ok(path: Path) -> Path:
    # чистая «нормальная» картинка без текста/лиц
    h, w = 900, 1200
    x = np.linspace(180, 220, w, dtype=np.uint8)
    img = np.repeat(x[None, :], h, axis=0)
    img = cv2.merge([img, img, img])
    cv2.imwrite(str(path), img)
    return path

# --- вспомогательное: поиск лиц в кэше проекта (если есть реальные фото) ---
def _scan_cache_for_face(limit: int = 200) -> str | None:
    img_dir = Path("_cache") / "images"
    if not img_dir.exists():
        return None
    cnt = 0
    for p in img_dir.glob("*.jpg"):
        try:
            img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if has_face(img):
                return str(p)
        finally:
            cnt += 1
            if cnt >= limit:
                break
    return None

def main():
    out = {
        "env": {},
        "unit": {},
        "integration": {},
        "notes": []
    }

    # 0) Пробинг OCR-бэкендов + состояние каскада лиц
    try:
        back = _f._probe_ocr_backends_once()  # {'paddle': bool, 'easyocr': bool}
    except Exception as e:
        back = {"error": f"{e.__class__.__name__}: {e}"}
    try:
        haar_ok = not _f._haar.empty()
    except Exception:
        haar_ok = False

    out["env"]["ocr_backends"] = back
    out["env"]["haar_loaded"] = haar_ok

    # 1) Юнит-тесты на синтетике
    d = _mk_dir()
    p_text = _img_text(d / "text.jpg")
    p_low  = _img_lowres(d / "low.jpg")
    p_bad  = _img_bad_aspect(d / "bad_aspect.jpg")
    p_ok   = _img_clean_ok(d / "clean.jpg")

    def _read(p: Path):
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

    img_text = _read(p_text)
    img_low  = _read(p_low)
    img_bad  = _read(p_bad)
    img_ok   = _read(p_ok)

    out["unit"]["has_text_positive"] = bool(has_text(img_text, min_words=1))
    out["unit"]["has_text_negative"] = bool(has_text(img_ok,   min_words=1))  # ожидаем False
    out["unit"]["is_lowres"]         = bool(is_lowres(img_low))
    out["unit"]["bad_aspect"]        = bool(bad_aspect(img_bad))
    out["unit"]["has_face_on_clean"] = bool(has_face(img_ok))  # ожидаем False

    # 2) Интеграция: pass_all_filters на каждом типе
    def _paf(p: Path):
        ok, reasons = pass_all_filters(str(p))
        return {"ok": bool(ok), "reasons": reasons}

    out["integration"]["text.jpg"]       = _paf(p_text)
    out["integration"]["low.jpg"]        = _paf(p_low)
    out["integration"]["bad_aspect.jpg"] = _paf(p_bad)
    out["integration"]["clean.jpg"]      = _paf(p_ok)

    # 3) Реальная проверка детектора лиц на кэше (если уже качали превью)
    face_path = _scan_cache_for_face()
    out["integration"]["face_in_cache_sample"] = face_path or None
    if face_path is None:
        out["notes"].append(
            "В _cache/images не найдено картинок с лицами (или кэша нет). "
            "Это НЕ ошибка. Для явной проверки положи любое фото с лицом в .selftest_filters/face.jpg и перезапусти."
        )
    else:
        out["notes"].append(f"Нашёл лицо в кэше: {face_path}")

    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
