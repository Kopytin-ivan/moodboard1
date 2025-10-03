from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal  

class PinCard(BaseModel):
    id: str
    preview_url: str
    source_url: str
    title: Optional[str] = None
    alt_text: Optional[str] = None

class MoodboardRequest(BaseModel):
    prompt: str
    limit: int = Field(24, ge=8, le=40)
    lang: str = "auto"

class CardOut(BaseModel):
    id: str
    preview_url: str
    source_url: str
    score: float = 0.0
    aesthetic: float = 0.0
    
class LayoutItem(BaseModel):
    i: int
    x: int
    y: int
    w: int
    h: int

class MoodboardResponse(BaseModel):
    prompt: str
    queries: List[str]
    cards: List[CardOut]
    palette: Optional[List[str]] = None             # HEX-палитра борда
    layout: Optional[List[LayoutItem]] = None       # ← новое поле
    filter_stats: Optional[Dict[str, int]] = None   # воронка фильтров


