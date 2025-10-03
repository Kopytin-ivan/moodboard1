# backend/config.py

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
BASE_DIR = Path(__file__).resolve().parent
# =========================
# Константы фильтров (вне Pydantic-модели!)
# =========================
MIN_SHORT_SIDE = 512
ASPECT_MIN = 0.5
ASPECT_MAX = 2.0

# --- детектор текста (для PaddleOCR/EasyOCR) ---
OCR_MIN_BOX_SCORE = 0.5            # минимальная уверенность детектора для бокса
OCR_MIN_BOX_AREA_FRAC = 0.003      # доля от площади изображения для «значимого» бокса (0.3%)
OCR_MIN_TOTAL_AREA_FRAC = 0.01     # суммарная доля площади боксов, после которой считаем «есть заметный текст» (1%)
OCR_MIN_SIGNIFICANT_BOXES = 2      # или хотя бы 2 крупных бокса

# --- совместимость со старым Tesseract (fallback/не обязателен) ---
OCR_LANG = "eng+rus"
OCR_MAX_WORDS = 3
OCR_MIN_CONF = 70

FACE_MODEL_PATH: str = "backend/models/centerface.onnx"  # можешь заменить на retinaface-tiny.onnx
FACE_CONF_THR: float = 0.5                 # порог уверенности для детектора лиц

# =========================
# Переменные окружения / настройки приложения
# =========================
class Settings(BaseSettings):
    # ---- ML / CLIP ----
    USE_CLIP: bool = False                 # включать CLIP-rerank (на старте False; можно True в .env)
    CLIP_MODEL: str = "ViT-B-32"           # см. embeds.py — ожидает такие имена
    CLIP_DEVICE: str = "cpu"               # "cuda" если есть CUDA (или оставь "cpu")

    # ---- Кэш ----
    CACHE_DIR: Path = Path.home() / ".moodboard_cache"
    CACHE_TTL_SEC: int = 3 * 24 * 3600     # 3 дня

    # ---- LLM ----
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.aitunnel.ru/v1"
    OPENAI_MODEL: str = "gpt-5-nano"

    # ---- Pinterest ----
    PINTEREST_API_KEY: str = ""
    PINTEREST_BASE_URL: str = "https://api-sandbox.pinterest.com"

    # ---- Apify ----
    USE_APIFY: bool = False
    APIFY_TOKEN: str = ""
    APIFY_ACTOR: str = "apify/web-scraper"
    APIFY_TIMEOUT: int = 60
    APIFY_PROXY_COUNTRY: str = "US"

    # ---- TLS/SSL ----
    DEV_ONLY_SKIP_SSL_VERIFY: bool = False
    PINTEREST_CA_BUNDLE: Optional[str] = None

    # ---- App ----
    APP_LANG: str = "ru"

    # ---- Limits / timeouts ----
    PIN_LIMIT_PER_QUERY: int = 60
    CANDIDATES_TARGET: int = 200
    TIMEOUT_SECONDS: int = 12

    # ---- Serial Pinterest switch ----
    PINTEREST_SERIAL: bool = True

    # === AESTHETIC / SCORING ===
    SCORE_ALPHA: float = 0.6       # вес семантики CLIP
    SCORE_BETA: float = 0.4        # вес эстетики LAION
    AESTHETIC_MIN: float = 5.5    # жёсткий отсев (включишь 5.5 позже при желании)

    # Папка с моделями (в т.ч. aesthetic/*.npy)
    MODELS_DIR: str = str((BASE_DIR / "models").resolve())


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )



# Глобальный инстанс настроек
settings = Settings()

# FS + логгер
settings.CACHE_DIR = Path(str(settings.CACHE_DIR)).expanduser()
(settings.CACHE_DIR / "images").mkdir(parents=True, exist_ok=True)
(settings.CACHE_DIR / "jobs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s: %(message)s")
log = logging.getLogger("moodboard")
log.info(f"[Settings] PINTEREST_BASE_URL={settings.PINTEREST_BASE_URL}")

