# backend/pipeline/rank.py
from __future__ import annotations
import numpy as np
from typing import List, Sequence
from PIL import Image
from imagehash import phash as _phash

# ---- TOTAL SCORE (пока без эстетики; оставим хук под неё) ----
def total_score(sim_to_prompt: float, aesth_0_10: float | None = None, penalties: dict | None = None) -> float:
    alpha, beta = 1.0, 0.0   # пока учитываем только CLIP-сходство; эстетика = позже
    penalty = 0.0
    if penalties:
        if penalties.get("maybe_text"): penalty += 0.10
        if penalties.get("maybe_face"): penalty += 0.20
        if penalties.get("maybe_logo"): penalty += 0.10
    aesth_term = (aesth_0_10 or 0.0) / 10.0
    return alpha*sim_to_prompt + beta*aesth_term - penalty

# ---- Dedup: pHash + близость эмбеддингов ----
def deduplicate(paths: Sequence[str], embeds: np.ndarray, max_phash_dist: int = 8, emb_sim_thr: float = 0.92) -> List[int]:
    keep: List[int] = []
    hashes: List = []

    # нормируем эмбеддинги заранее (косинус = скаляр после L2-нормы)
    X = embeds.astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    for i, p in enumerate(paths):
        # phash
        try:
            h = _phash(Image.open(p))
        except Exception:
            h = None

        is_dup = False

        # по pHash
        if h is not None:
            for h2 in hashes:
                if h - h2 <= max_phash_dist:
                    is_dup = True
                    break

        # по эмбеддингам (косинусная близость)
        if not is_dup:
            for j in keep:
                if float(np.dot(X[i], X[j])) >= emb_sim_thr:
                    is_dup = True
                    break

        if not is_dup:
            keep.append(i)
            if h is not None:
                hashes.append(h)

    return keep  # индексы оставленных (в исходном порядке)

# ---- MMR-диверсификация ----
def pairwise_cosine(embeds: np.ndarray) -> np.ndarray:
    X = embeds.astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X @ X.T  # (N,N)

def mmr_select(sim_to_query: np.ndarray, sim_matrix: np.ndarray, k: int = 24, lam: float = 0.7) -> List[int]:
    selected: List[int] = []
    cand = set(range(len(sim_to_query)))
    while len(selected) < k and cand:
        best, best_val = None, -1e9
        for i in cand:
            div = 0.0 if not selected else max(sim_matrix[i, j] for j in selected)
            val = lam*float(sim_to_query[i]) - (1.0 - lam)*div
            if val > best_val:
                best, best_val = i, val
        selected.append(best)
        cand.remove(best)
    return selected
