# backend/pipeline/aesthetic.py
from __future__ import annotations
import os
import math
import json
import logging
from typing import List, Optional

import numpy as np
from PIL import Image, ImageStat, ImageFilter

log = logging.getLogger(__name__)

"""
AestheticScorer
---------------
Реализует два режима:
1) 'linear_head' — линейная голова LAION поверх CLIP-эмбеддингов.
   Требуются веса .npy (вектор размерности под эмбеддинг CLIP).
   Файлы:
     - models/aesthetic/vit_l14_linear.npy     (dim=768)
     - models/aesthetic/vit_b32_linear.npy     (dim=512)
2) 'heuristic' — запасной вариант без весов (работает всегда).
   Эвристика по яркости/контрасту/резкости → возвращает 0..10.

Сигнатуры:
 - score_from_embeddings(embeds: np.ndarray, arch: str) -> List[float]
 - score_from_images(paths: List[str]) -> List[float]   # эвристика
"""

def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

class AestheticScorer:
    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self.w_cache: dict[str, np.ndarray] = {}

    def _load_linear_weights(self, arch: str) -> Optional[np.ndarray]:
        # arch: 'vit_l14' | 'vit_b32'
        fname = {
            "vit_l14": "vit_l14_linear.npy",
            "vit_b32": "vit_b32_linear.npy",
        }.get(arch)
        if not fname:
            return None
        path = os.path.join(self.models_dir, "aesthetic", fname)
        if not os.path.exists(path):
            log.warning("[Aesthetic] linear weights not found: %s", path)
            return None
        try:
            if path in self.w_cache:
                return self.w_cache[path]
            w = np.load(path).astype(np.float32)
            # Нормализуем на всякий случай
            if w.ndim > 1:
                w = w.reshape(-1)
            self.w_cache[path] = w
            log.info("[Aesthetic] loaded linear weights: %s (dim=%d)", path, w.shape[0])
            return w
        except Exception as e:
            log.exception("[Aesthetic] failed to load weights: %s", e)
            return None

    # ------- ПУТЬ 1: LAION-совместимая линейная голова -------
    def score_from_embeddings(self, embeds: np.ndarray, arch: str) -> List[float]:
        """
        embeds: (N, D) эмбеддинги CLIP (уже L2-нормированные).
        arch: 'vit_l14' или 'vit_b32' — должен соответствовать выбранной модели CLIP.
        Возвращает список оценок 0..10.
        """
        if embeds is None or len(embeds) == 0:
            return []

        w = self._load_linear_weights(arch)
        if w is None:
            log.warning("[Aesthetic] fallback to heuristic (no linear weights)")
            # деградируем на эвристику (масштабируем под 0..10)
            return self._heuristic_from_embeds_fallback(embeds)

        if embeds.ndim != 2:
            raise ValueError("embeds must be (N, D)")
        if embeds.shape[1] != w.shape[0]:
            log.warning("[Aesthetic] dim mismatch: embeds=%d vs w=%d; using heuristic",
                        embeds.shape[1], w.shape[0])
            return self._heuristic_from_embeds_fallback(embeds)

        # Линейная модель: score = sigmoid(<emb, w>) * 10
        logits = embeds @ w
        aesthetics = (_safe_sigmoid(logits) * 10.0).astype(np.float32)
        return aesthetics.tolist()

    def _heuristic_from_embeds_fallback(self, embeds: np.ndarray) -> List[float]:
        # Если нет весов — вернём умеренно-полезный скор:
        # больше разнообразия/энергии (некоторый прокси по L2) -> немного выше.
        # Диапазон ~ 4..7
        if embeds.size == 0:
            return []
        norms = np.clip(np.linalg.norm(embeds, axis=1), 0.5, 2.0)
        norms = (norms - norms.min()) / max(1e-6, (norms.max() - norms.min()))
        return (4.0 + 3.0 * norms).tolist()

    # ------- ПУТЬ 2: Эвристика по пикселям (работает всегда) -------
    def score_from_images(self, paths: List[str]) -> List[float]:
        """
        Эвристическая оценка 0..10 по пиксельным свойствам:
          - усреднённая контрастность (std яркости)
          - умеренная яркость
          - резкость (variance Laplacian)
        Не заменяет LAION, но даёт адекватный бэкап.
        """
        out: List[float] = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                # Яркость
                stat = ImageStat.Stat(img.convert("L"))
                mean = stat.mean[0]      # 0..255
                std = stat.stddev[0]     # 0..~80

                # Резкость: variance of Laplacian (приближение)
                lap = img.convert("L").filter(ImageFilter.FIND_EDGES)
                s = ImageStat.Stat(lap).var[0]  # чем выше, тем резче

                # Нормируем в грубые 0..1
                mean_n = 1.0 - abs(mean - 128.0) / 128.0      # лучше средняя яркость
                std_n  = min(std / 64.0, 1.0)                  # умеренная контрастность
                sharp_n= min(s / 500.0, 1.0)                   # грубая шкала

                score01 = 0.4*mean_n + 0.3*std_n + 0.3*sharp_n
                out.append(float(10.0 * np.clip(score01, 0.0, 1.0)))
            except Exception:
                out.append(5.0)  # нейтральный fallback
        return out
