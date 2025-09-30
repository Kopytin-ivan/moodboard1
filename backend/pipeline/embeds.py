# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
import open_clip
from ..config import settings

log = logging.getLogger("moodboard")


@lru_cache(maxsize=1)
def _load_clip():
    """
    Ленивая инициализация модели OpenCLIP + препроцессинг + токенайзер.
    Поднимается один раз на процесс.
    """
    # Выбор устройства
    device = "cuda" if (getattr(settings, "CLIP_DEVICE", "cpu") == "cuda" and torch.cuda.is_available()) else "cpu"
    model_name = getattr(settings, "CLIP_MODEL", "ViT-B-32")

    # Загрузка модели и трансформов
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained="openai",   # можно заменить на нужный чекпойнт
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    log.info(f"[CLIP] загружен {model_name} на {device}")
    return model, preprocess, tokenizer, device


def _use_clip() -> bool:
    return bool(getattr(settings, "USE_CLIP", False))


def _safe_open_image(path: str) -> Image.Image:
    """
    Надёжное открытие изображений: RGB, без утечек дескрипторов.
    При ошибке возвращает серый плейсхолдер.
    """
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (127, 127, 127))


@torch.inference_mode()
def _encode_text(prompt: str, device: str) -> torch.Tensor:
    model, _, tokenizer, _ = _load_clip()
    toks = tokenizer([prompt]).to(device)
    text_feat = model.encode_text(toks)              # (1, D)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat  # (1, D)


@torch.inference_mode()
def _encode_images(paths: List[str], device: str, batch_size: int = 16) -> torch.Tensor:
    model, preprocess, _, _ = _load_clip()
    feats = []
    N = len(paths)
    for s in range(0, N, batch_size):
        batch_paths = paths[s : s + batch_size]
        imgs = [_safe_open_image(p) for p in batch_paths]
        batch = torch.stack([preprocess(im) for im in imgs], dim=0).to(device)  # (B, C, H, W)
        image_feat = model.encode_image(batch)                                  # (B, D)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        feats.append(image_feat.cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty((0, 1))


def clip_rerank(
    prompt: str,
    image_paths: List[str],
    top_k: int,
    return_scores: bool = False,
    batch_size: int = 16,
) -> List[int] | Tuple[List[int], List[float]]:
    """
    Отсортировать картинки по релевантности тексту (OpenCLIP).
    Возвращает индексы (и опционально скоры) по убыванию.

    Если CLIP отключён/недоступен — вернёт исходный порядок (0..N-1) срезом top_k.
    """
    N = len(image_paths)
    if N == 0:
        return ([], []) if return_scores else []

    if not _use_clip():
        idx = list(range(min(top_k, N)))
        return (idx, [0.0] * len(idx)) if return_scores else idx

    try:
        model, _, _, device = _load_clip()

        text_feat = _encode_text(prompt, device)                 # (1, D)
        image_feats = _encode_images(image_paths, device, batch_size)  # (N, D)
        if image_feats.shape[0] == 0:
            idx = list(range(min(top_k, N)))
            return (idx, [0.0] * len(idx)) if return_scores else idx

        # cosine sim: (1,D) @ (D,N) -> (1,N)
        sims = (text_feat.cpu() @ image_feats.T).squeeze(0)      # (N,)
        order = torch.argsort(sims, descending=True).tolist()
        order = order[: min(top_k, N)]
        if return_scores:
            scores = [float(sims[i]) for i in order]
            return order, scores
        return order
    except Exception as e:
        log.warning(f"[CLIP] rerank failed: {e}")
        idx = list(range(min(top_k, N)))
        return (idx, [0.0] * len(idx)) if return_scores else idx
    
def clip_image_embeddings(paths: list[str]) -> np.ndarray:
    """
    Возвращает np.ndarray формы (N, D) — L2-нормированные эмбеддинги изображений.
    Использует уже объявленный _load_clip().
    """
    model, preprocess, _, device = _load_clip()
    embs: list[np.ndarray] = []

    for p in paths:
        try:
            img = _safe_open_image(p)
            x = preprocess(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                e = model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.squeeze(0).detach().cpu().numpy().astype(np.float32))
        except Exception:
            # длину D берём у текстового энкодера: прогоняем «пустую» токенизацию один раз
            try:
                _, _, tokenizer, device2 = _load_clip()
                toks = tokenizer([""])
                with torch.inference_mode():
                    t = model.encode_text(toks.to(device2))
                    D = int(t.shape[-1])
            except Exception:
                D = 512  # безопасный дефолт
            embs.append(np.zeros((D,), dtype=np.float32))

    return np.stack(embs, axis=0) if embs else np.zeros((0, 512), dtype=np.float32)


def clip_text_similarity(prompt: str, image_embeds: np.ndarray) -> np.ndarray:
    """
    Косинусная близость текста к эмбеддингам картинок (предполагаем, что image_embeds уже L2-нормированы).
    """
    model, _, tokenizer, device = _load_clip()
    with torch.inference_mode():
        toks = tokenizer([prompt]).to(device)
        t = model.encode_text(toks)
        t = (t / t.norm(dim=-1, keepdim=True)).detach().cpu().numpy().reshape(-1)  # (D,)
    if image_embeds.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (image_embeds @ t).astype(np.float32)
