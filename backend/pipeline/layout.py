# backend/pipeline/layout.py
from __future__ import annotations
from typing import List, Dict

# Пресеты: количество колонок × строк
_PRESETS = {
    20: (5, 4),
    24: (6, 4),
    36: (6, 6),
}

def auto_layout(n: int, preset: int = 24) -> List[Dict]:
    """
    Возвращает список тайлов вида:
    {"i": <index>, "x": int, "y": int, "w": int, "h": int}
    где w=h=1 (простая сетка), координаты в ячейках.
    Первые 2 карточки делаем "героями" (w=2,h=2), если помещаются.
    """
    cols, rows = _PRESETS.get(preset, (6, 4))
    # если n > вместимости — расширим строки
    cap = cols * rows
    while n > cap:
        rows += 1
        cap = cols * rows

    tiles: List[Dict] = []
    grid = [[False] * cols for _ in range(rows)]

    def place(w: int, h: int) -> tuple[int, int] | None:
        for y in range(rows - h + 1):
            for x in range(cols - w + 1):
                ok = True
                for yy in range(y, y+h):
                    for xx in range(x, x+w):
                        if grid[yy][xx]:
                            ok = False; break
                    if not ok: break
                if ok:
                    for yy in range(y, y+h):
                        for xx in range(x, x+w):
                            grid[yy][xx] = True
                    return x, y
        return None

    # Герои
    hero_count = 2 if n >= 2 and cols >= 4 and rows >= 2 else 0
    used = 0
    for i in range(hero_count):
        pos = place(2, 2)
        if pos is None:
            break
        x, y = pos
        tiles.append({"i": used, "x": x, "y": y, "w": 2, "h": 2})
        used += 1

    # Остальные
    while used < n:
        pos = place(1, 1)
        if pos is None:
            # если вдруг переполнено — добавим строку
            rows += 1
            grid.append([False] * cols)
            continue
        x, y = pos
        tiles.append({"i": used, "x": x, "y": y, "w": 1, "h": 1})
        used += 1

    # Отсортируем по (y,x)
    tiles.sort(key=lambda t: (t["y"], t["x"]))
    # Пронумеруем заново поле "i" по порядку
    for idx, t in enumerate(tiles):
        t["i"] = idx
    return tiles
