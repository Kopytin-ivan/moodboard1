from __future__ import annotations
import cv2, numpy as np
from pathlib import Path
from PIL import Image
from ..config import (
    MIN_SHORT_SIDE, ASPECT_MIN, ASPECT_MAX,
    OCR_LANG, OCR_MAX_WORDS, OCR_MIN_CONF,
)

from ..config import (
    MIN_SHORT_SIDE, ASPECT_MIN, ASPECT_MAX,
    OCR_LANG, OCR_MAX_WORDS, OCR_MIN_CONF,
    OCR_MIN_BOX_SCORE, OCR_MIN_BOX_AREA_FRAC, OCR_MIN_TOTAL_AREA_FRAC, OCR_MIN_SIGNIFICANT_BOXES,
)

# ---------- утилиты ----------
def _read_bgr(path: str) -> np.ndarray:
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
_paddle_det = None
_easyocr_reader = None

def _get_paddle_detector():
    global _paddle_det
    if _paddle_det is None:
        try:
            from paddleocr import PaddleOCR
            # ВАЖНО: rec=False — только детектор; язык не влияет на детектор
            _paddle_det = PaddleOCR(
                use_angle_cls=False, det=True, rec=False, lang='en', show_log=False
            )
        except Exception:
            _paddle_det = False  # помечаем как «не доступен»
    return _paddle_det

def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            # детектор CRAFT + CRNN; для детекции хватит lang=['en','ru']
            _easyocr_reader = easyocr.Reader(['en','ru'], gpu=False)
        except Exception:
            _easyocr_reader = False
    return _easyocr_reader

def _boxes_to_mask_metrics(img_shape, boxes, scores=None):
    """Считает сколько «значимых» боксов и их суммарную площадь (фракция от изображения)."""
    H, W = img_shape[:2]
    min_area = OCR_MIN_BOX_AREA_FRAC * (W * H)
    significant = 0
    total_area = 0.0
    for i, box in enumerate(boxes):
        # box может быть [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] (Paddle) или [x1,y1,x2,y2] (EasyOCR)
        if isinstance(box[0], (list, tuple)) and len(box) == 4:  # PaddleOCR poly
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:  # EasyOCR bbox
            x1, y1, x2, y2 = box

        w = max(0, x2 - x1); h = max(0, y2 - y1)
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

def has_text(img_bgr: np.ndarray) -> bool:
    """Новый детектор: сначала детект, потом простые пороги на площадь/кол-во блоков."""
    # 1) Пытаемся использовать PaddleOCR (детектор DB)
    det = _get_paddle_detector()
    if det:
        try:
            # PaddleOCR ожидает BGR/RGB путь или матрицу; отдаём матрицу напрямую
            # Результат: список списков; каждый элемент: [ [poly 4 точки], score ]
            res = det.ocr(img_bgr, det=True, rec=False)
            # res может быть [[]] или список по батчу; нормализуем
            polys, scores = [], []
            if isinstance(res, list):
                # формат v2.7: res[0] -> список детекций
                items = res[0] if len(res) > 0 else []
                for it in items:
                    if isinstance(it, list) and len(it) >= 2:
                        poly, sc = it[0], it[1]
                        polys.append(poly); scores.append(sc)
            sig, total = _boxes_to_mask_metrics(img_bgr.shape, polys, scores)
            if sig >= OCR_MIN_SIGNIFICANT_BOXES or total >= OCR_MIN_TOTAL_AREA_FRAC:
                return True
            return False
        except Exception:
            pass  # упадём на fallback

    # 2) Fallback на EasyOCR (детектор CRAFT)
    reader = _get_easyocr()
    if reader:
        try:
            # returns: [(bbox, text, conf), ...]; bbox = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            res = reader.readtext(img_bgr, detail=1, paragraph=False)  # CPU
            polys = [r[0] for r in res]
            scores = [float(r[2]) for r in res]
            sig, total = _boxes_to_mask_metrics(img_bgr.shape, polys, scores)
            if sig >= OCR_MIN_SIGNIFICANT_BOXES or total >= OCR_MIN_TOTAL_AREA_FRAC:
                return True
            return False
        except Exception:
            pass

    # 3) Последний резерв — старый Tesseract (как было), но с защитами от FP
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    cfg = f'--oem 1 --psm 6 -l {OCR_LANG}'
    data = pytesseract.image_to_data(thr, config=cfg, output_type=pytesseract.Output.DICT)

    if not data or "conf" not in data or "text" not in data:
        return False

    H, W = img_bgr.shape[:2]
    min_area = OCR_MIN_BOX_AREA_FRAC * (W * H)
    words = 0
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
            bw, bh = int(data["width"][i]), int(data["height"][i])
        except Exception:
            conf, bw, bh = -1, 0, 0

        if conf < OCR_MIN_CONF:
            continue
        if len(txt) < 3 or txt.isdigit():
            continue
        if bw * bh < min_area:
            continue
        words += 1
        if words >= OCR_MIN_SIGNIFICANT_BOXES:
            return True

    return False


# ---------- лица (бюджетный baseline на CPU) ----------
# Временно используем Haar-каскад OpenCV; затем можно заменить на CenterFace-ONNX.
_haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def has_face(img_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
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

#--------------------------------------------TEST-----------------
from dataclasses import dataclass
from typing import List

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
