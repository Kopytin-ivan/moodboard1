from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image  # noqa: F401

from ..config import (
    MIN_SHORT_SIDE, ASPECT_MIN, ASPECT_MAX,
    OCR_LANG, OCR_MAX_WORDS, OCR_MIN_CONF,
    OCR_MIN_BOX_SCORE, OCR_MIN_BOX_AREA_FRAC, OCR_MIN_TOTAL_AREA_FRAC, OCR_MIN_SIGNIFICANT_BOXES,
)

# --- Тише низкоуровневые логи от Paddle/PP, если они присутствуют
os.environ.setdefault("FLAGS_minloglevel", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

# --- Логгер этого модуля
logger = logging.getLogger("moodboard.filters")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[filters] %(levelname)s: %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# --- Глобальные OCR инстансы (ленивая инициализация)
_paddle_ocr: object | bool | None = None  # True-подобный объект -> инстанс, False -> отключить дальнейшие попытки
_easyocr_reader: object | bool | None = None

# --- Флаги однократного пробинга
_OCR_PROBED = False
_OCR_BACKENDS = {"paddle": False, "easyocr": False}


def _probe_ocr_backends_once() -> dict:
    """Пробуем импортировать OCR-бэкенды один раз и логируем, что вообще доступно в окружении."""
    global _OCR_PROBED, _OCR_BACKENDS
    if _OCR_PROBED:
        return _OCR_BACKENDS

    # PaddleOCR
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        _OCR_BACKENDS["paddle"] = True
    except Exception as e:
        logger.info(f"PaddleOCR недоступен: {e.__class__.__name__}: {e}")

    # EasyOCR
    try:
        import easyocr  # noqa: F401
        _OCR_BACKENDS["easyocr"] = True
    except Exception as e:
        logger.info(f"EasyOCR недоступен: {e.__class__.__name__}: {e}")

    _OCR_PROBED = True
    logger.info(f"OCR backends: paddle={_OCR_BACKENDS['paddle']} easyocr={_OCR_BACKENDS['easyocr']}")
    return _OCR_BACKENDS


def _get_paddle_ocr():
    """
    Ленивая инициализация полноценного PaddleOCR (det+rec).
    ВАЖНО: не передаём неподдерживаемые аргументы (например, show_log).
    """
    global _paddle_ocr
    if _paddle_ocr is False:
        return None
    if _paddle_ocr is not None:
        return _paddle_ocr

    try:
        from paddleocr import PaddleOCR  # type: ignore
        # Язык возьмём из конфигурации (OCR_LANG). Для нашей задачи достаточно 'en' или 'ru'.
        lang_raw = (OCR_LANG or "").lower()
        lang = "ru" if "ru" in lang_raw else "en"  # Paddle ждёт один язык

        _paddle_ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,     # например, 'en' или 'ru'
            # show_log НЕЛЬЗЯ передавать — в твоей версии PaddleOCR его нет
        )
        logger.info(f"PaddleOCR: initialized (lang={lang}, use_angle_cls=True)")
    except Exception as e:
        logger.info(f"PaddleOCR недоступен при инициализации: {e.__class__.__name__}: {e}")
        _paddle_ocr = False
        return None
    return _paddle_ocr


def _get_easyocr():
    """Ленивая инициализация EasyOCR (детектор+рекогнайзер)."""
    global _easyocr_reader
    if _easyocr_reader is False:
        return None
    if _easyocr_reader is not None:
        return _easyocr_reader

    try:
        import easyocr  # type: ignore
        # Всегда поддерживаем и 'en', и 'ru', независимо от OCR_LANG
        langs = ["en", "ru"]
        _easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)
        logger.info(f"EasyOCR: initialized (langs={langs}, gpu=False)")
    except Exception as e:
        logger.info(f"EasyOCR недоступен при инициализации: {e.__class__.__name__}: {e}")
        _easyocr_reader = False
        return None
    return _easyocr_reader

# ---------- утилиты ----------
def _read_bgr(path: str) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    # imdecode -> корректно открывает пути с кириллицей
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def is_lowres(img_bgr: np.ndarray) -> bool:
    h, w = img_bgr.shape[:2]
    return min(h, w) < MIN_SHORT_SIDE


def bad_aspect(img_bgr: np.ndarray) -> bool:
    h, w = img_bgr.shape[:2]
    r = w / max(1, h)
    return not (ASPECT_MIN <= r <= ASPECT_MAX)


# ---------- текст/логотипы ----------

def _boxes_to_mask_metrics(img_shape, boxes, scores=None):
    """Считает сколько «значимых» боксов и их суммарную площадь (фракция от изображения)."""
    H, W = img_shape[:2]
    min_area = OCR_MIN_BOX_AREA_FRAC * (W * H)
    significant = 0
    total_area = 0.0
    for i, box in enumerate(boxes):
        # box может быть [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (Paddle) или [x1,y1,x2,y2] (EasyOCR)
        if isinstance(box[0], (list, tuple)) and len(box) == 4:  # PaddleOCR poly
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:  # EasyOCR bbox
            x1, y1, x2, y2 = box

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        area = w * h
        if area <= 0:
            continue

        # отфильтруем супер-длинные «спички» (линии на планах)
        ar = max(w / max(1, h), h / max(1, w))
        if ar > 15:
            continue

        # учёт уверенности детектора
        sc_ok = True
        if scores is not None:
            try:
                sc_ok = float(scores[i]) >= OCR_MIN_BOX_SCORE
            except Exception:
                sc_ok = True

        if sc_ok and area >= min_area:
            significant += 1
            total_area += area

    total_frac = total_area / float(W * H)
    return significant, total_frac


def has_text(
    img_bgr: np.ndarray,
    min_words: int = 3,
    min_confidence: float = 0.60,
    languages: Tuple[str, ...] = ("ru", "en"),
) -> bool:
    """
    Возвращает True, если в изображении обнаружен заметный текст/логотип.
    Критерии:
      1) найдено >= min_words слов (по уверенности);
      2) ИЛИ выполнен порог по «значимым» боксам и общей площади текста.
    """
    if img_bgr is None or img_bgr.size == 0:
        logger.info("has_text: пустое изображение -> False")
        return False

    # Масштабируем до 1024 по длинной стороне
    h, w = img_bgr.shape[:2]
    scale = 1024.0 / max(h, w)
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Лёгкая предобработка
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 15)
    prep = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    backends = _probe_ocr_backends_once()

    
    # ---------- 1) EasyOCR ----------
    if backends.get("easyocr"):
        logger.info("has_text: использую OCR=EasyOCR")
        reader = _get_easyocr()
        if reader is not None:
            try:
                # Формат: [(bbox(x1,y1,x2,y2), text, conf), ...]
                result = reader.readtext(prep)

                words = []
                boxes = []
                scores = []
                for (bbox, text, conf) in result:
                    if text and text.strip() and (conf is None or conf >= min_confidence):
                        words.append(text.strip())
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                    else:
                        # EasyOCR иногда отдаёт 4-точечный полигон
                        try:
                            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                        except Exception:
                            continue
                    boxes.append([x1, y1, x2, y2])
                    scores.append(conf if conf is not None else 1.0)

                if len(words) >= min_words:
                    logger.info(f"has_text[EasyOCR]: слов={len(words)} (порог={min_words}) -> True")
                    return True

                significant, total_frac = _boxes_to_mask_metrics(img_bgr.shape, boxes, scores)
                logger.info(
                    f"has_text[EasyOCR]: boxes_sig={significant} "
                    f"total_frac={total_frac:.4f} "
                    f"(min_boxes={OCR_MIN_SIGNIFICANT_BOXES}, min_frac={OCR_MIN_TOTAL_AREA_FRAC})"
                )
                if significant >= OCR_MIN_SIGNIFICANT_BOXES and total_frac >= OCR_MIN_TOTAL_AREA_FRAC:
                    return True

            except Exception as e:
                logger.info(f"has_text: EasyOCR ошибка: {e.__class__.__name__}: {e}")

    # ---------- 2) PaddleOCR ----------
    if backends.get("paddle"):
        logger.info("has_text: использую OCR=PaddleOCR")
        ocr = _get_paddle_ocr()  # ВАЖНО: без show_log
        if ocr is not None:
                    try:
                        # Paddle ожидает 3-канальный BGR (наш prep — серый)
                        inp_bgr = cv2.cvtColor(prep, cv2.COLOR_GRAY2BGR)

                        # Поддержим разные версии API: со старым и без параметра cls
                        try:
                            raw = ocr.ocr(inp_bgr, cls=True)
                        except TypeError:
                            raw = ocr.ocr(inp_bgr)

                        # У некоторых версий сверху есть батч-обёртка: [[(poly, ...), ...]]
                        items = raw[0] if (isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list)) else raw

                        words: List[str] = []
                        boxes: List[List[float]] = []
                        scores: List[float] = []

                        for it in items:
                            if not isinstance(it, (list, tuple)) or len(it) < 2:
                                continue

                            poly = it[0]
                            text = None
                            conf = None

                            # Вариант A: (poly, (text, conf))
                            if len(it) >= 2 and isinstance(it[1], (list, tuple)) and len(it[1]) >= 2 and isinstance(it[1][0], str):
                                text, conf = it[1][0], float(it[1][1])

                            # Вариант B: (poly, text, conf)
                            elif len(it) >= 3 and isinstance(it[1], str):
                                text, conf = it[1], float(it[2])

                            # Вариант C: (poly, {dict c полями})
                            elif len(it) >= 2 and isinstance(it[1], dict):
                                rec = it[1]
                                text = rec.get("text") or rec.get("label") or ""
                                conf = float(rec.get("score") or rec.get("confidence") or 1.0)

                            if not text:
                                continue

                            if conf is None or conf >= min_confidence:
                                words.append(text.strip())

                            # Преобразуем poly -> bbox для метрик площади
                            try:
                                if isinstance(poly, (list, tuple)) and len(poly) == 4 and not isinstance(poly[0], (list, tuple)):
                                    # Уже (x1, y1, x2, y2)
                                    x1, y1, x2, y2 = poly
                                else:
                                    xs = [p[0] for p in poly]
                                    ys = [p[1] for p in poly]
                                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                                boxes.append([x1, y1, x2, y2])
                                scores.append(conf if conf is not None else 1.0)
                            except Exception:
                                pass

                        if len(words) >= min_words:
                            logger.info(f"has_text[PaddleOCR]: слов={len(words)} (порог={min_words}) -> True")
                            return True

                        significant, total_frac = _boxes_to_mask_metrics(img_bgr.shape, boxes, scores)
                        logger.info(
                            f"has_text[PaddleOCR]: boxes_sig={significant} total_frac={total_frac:.4f} "
                            f"(min_boxes={OCR_MIN_SIGNIFICANT_BOXES}, min_frac={OCR_MIN_TOTAL_AREA_FRAC})"
                        )
                        if significant >= OCR_MIN_SIGNIFICANT_BOXES and total_frac >= OCR_MIN_TOTAL_AREA_FRAC:
                            return True

                    except Exception as e:
                        logger.info(f"has_text: PaddleOCR ошибка: {e.__class__.__name__}: {e} -> пробуем EasyOCR")



    # ---------- 3) Нет OCR ----------
    logger.info("has_text: текст/логотип не обнаружен -> False")
    return False



# ---------- лица (бюджетный baseline на CPU) ----------
# Временно используем Haar-каскад OpenCV; затем можно заменить на CenterFace-ONNX.
_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def has_face(img_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(80, 80))
    return len(faces) > 0


# ---------- композитная проверка ----------
def pass_all_filters(path: str) -> tuple[bool, dict]:
    reasons = {}
    img = _read_bgr(path)
    if img is None:
        reasons["io"] = True
        return False, reasons
    if is_lowres(img):
        reasons["lowres"] = True
        return False, reasons
    if bad_aspect(img):
        reasons["aspect"] = True
        return False, reasons
    if has_face(img):
        reasons["face"] = True
        return False, reasons
    if has_text(img):
        reasons["text"] = True
        return False, reasons
    return True, reasons


# ---------- диагностический отчёт ----------
@dataclass
class FilterReport:
    path: str
    has_face: bool
    has_text: bool
    is_lowres: bool
    bad_aspect: bool
    pass_all: bool
    reasons: List[str]


def analyze_image(path: str) -> FilterReport:
    """Диагностическая проверка одного файла из кэша."""
    img = _read_bgr(path)  # BGR
    if img is None:
        return FilterReport(
            path=path, has_face=False, has_text=False,
            is_lowres=True, bad_aspect=False,
            pass_all=False, reasons=["io"]
        )

    hf = has_face(img)
    ht = has_text(img)
    lr = is_lowres(img)
    ba = bad_aspect(img)

    reasons: List[str] = []
    if hf: reasons.append("face")
    if ht: reasons.append("text/logo")
    if lr: reasons.append("lowres")
    if ba: reasons.append("bad_aspect")

    return FilterReport(
        path=path, has_face=hf, has_text=ht,
        is_lowres=lr, bad_aspect=ba,
        pass_all=not (hf or ht or lr or ba),
        reasons=reasons
    )
