# backend/pipeline/palette.py
from __future__ import annotations
from typing import List
import cv2, numpy as np
from sklearn.cluster import KMeans

def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(x) for x in rgb]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def board_palette(image_paths: List[str], k: int = 6, samples_per_image: int = 6000) -> List[str]:
    """
    Собираем палитру из набора изображений:
    - сэмплим пиксели
    - переводим в Lab
    - кластеризация KMeans
    - возвращаем HEX-цвета центров, отсортированных по массе
    """
    pixels_lab = []

    for path in image_paths:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None: 
                continue
            # уменьшаем до макс. 1024 по длинной стороне для скорости
            h, w = img.shape[:2]
            scale = 1024.0 / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

            # BGR -> RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # сэмплинг
            H, W = rgb.shape[:2]
            N = min(samples_per_image, H*W)
            ys = np.random.randint(0, H, size=N)
            xs = np.random.randint(0, W, size=N)
            samples = rgb[ys, xs, :].astype(np.float32)

            # RGB -> Lab
            lab = cv2.cvtColor(samples.reshape(-1,1,3), cv2.COLOR_RGB2LAB).reshape(-1,3)
            pixels_lab.append(lab)
        except Exception:
            continue

    if not pixels_lab:
        return []

    X = np.concatenate(pixels_lab, axis=0)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X)
    centers_lab = kmeans.cluster_centers_

    # сортировка по массе кластера
    masses = np.bincount(labels, minlength=k).astype(np.float32)
    order = np.argsort(-masses)

    # Lab -> RGB -> HEX
    hexes = []
    for idx in order:
        lab = centers_lab[idx].reshape(1,1,3).astype(np.float32)
        rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB).reshape(3)
        rgb = np.clip(rgb, 0, 255)
        hexes.append(_rgb_to_hex(rgb))
    return hexes
