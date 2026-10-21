"""Небо RP Sky: тусклая серая дымка, за которой высоко над головой видны плиты потолка — бетон и металл.

Каждый пиксель считается по направлению взгляда, поэтому шесть граней куба стыкуются без швов. Грани описаны
так же, как в client/SkyRenderer.java (центр, вправо, вниз по картинке) — менять только вместе.

    python scripts/gen_sky.py            # грани в src/main/resources/assets/rpsky/textures/sky
    python scripts/gen_sky.py --preview  # ещё и превью «взгляд вверх» в build/sky_preview.png
"""
import argparse
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "src", "main", "resources", "assets", "rpsky", "textures", "sky")

SIZE = 1024
SS = 3  # сглаживание: SS×SS лучей на пиксель

# Грань: центр, «вправо» (u), «вниз» (v). Координаты Minecraft: +Y вверх, −Z север, +X восток.
FACES = {
    "up": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "down": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "north": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
    "south": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    "east": ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "west": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
}

# ---------------------------------------------------------------- параметры вида
PANEL = 0.34  # сторона плиты на потолке высотой 1 (над головой плита ≈ 19°)
GIRDER_EVERY = 4  # балка каждые N плит
SEAM = 0.022  # полуширина шва, доли плиты
GIRDER = 0.05  # полуширина балки, доли плиты
ZENITH_VIS = 0.26  # насколько видны плиты прямо над головой (0 — сплошная дымка)
HAZE_K = 1.35  # как быстро дымка густеет к горизонту
METAL_SHARE = 0.3

HAZE_TOP = np.array([0.53, 0.54, 0.55])
HAZE_HORIZON = np.array([0.63, 0.63, 0.62])
HAZE_BELOW = np.array([0.45, 0.45, 0.45])
CONCRETE = np.array([0.40, 0.39, 0.37])
METAL = np.array([0.31, 0.33, 0.36])
GIRDER_COL = np.array([0.25, 0.26, 0.27])
SEAM_COL = np.array([0.24, 0.24, 0.24])

MASK = np.uint64(0xFFFFFFFF)


def hash2(ix, iz, seed):
    """Случайное 0–1 для целой точки решётки."""
    h = (ix.astype(np.int64).astype(np.uint64) * np.uint64(374761393)
         + iz.astype(np.int64).astype(np.uint64) * np.uint64(668265263)
         + np.uint64(seed) * np.uint64(1442695041)) & MASK
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & MASK
    h = h ^ (h >> np.uint64(16))
    return h.astype(np.float64) / 4294967296.0


def value_noise(x, z, seed):
    ix, iz = np.floor(x), np.floor(z)
    fx, fz = x - ix, z - iz
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)
    a = hash2(ix, iz, seed)
    b = hash2(ix + 1, iz, seed)
    c = hash2(ix, iz + 1, seed)
    d = hash2(ix + 1, iz + 1, seed)
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uz


def fbm(x, z, seed, octaves=4):
    s, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        s = s + value_noise(x, z, seed + o * 17) * amp
        norm += amp
        x, z, amp = x * 2.03, z * 2.03, amp * 0.5
    return s / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ceiling(px, pz):
    """Цвет потолка в точке плоскости (px, pz)."""
    gz = pz / PANEL
    # Ряды плит сдвинуты друг относительно друга — не офисная сетка.
    gx = px / PANEL + hash2(np.floor(gz), np.zeros_like(gz), 9) * 0.8
    ix, iz = np.floor(gx), np.floor(gz)
    fx, fz = gx - ix, gz - iz
    r = hash2(ix, iz, 1)
    metal = hash2(ix, iz, 2) < METAL_SHARE
    shade = (hash2(ix, iz, 3) - 0.5) * 0.22

    # Швы: края плиты, у части плит — дополнительный шов посередине (плита из двух или четырёх).
    e = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fz, 1 - fz))
    split2 = r < 0.30
    split4 = (r >= 0.30) & (r < 0.42)
    e = np.where(split2 | split4, np.minimum(e, np.abs(fx - 0.5)), e)
    e = np.where(split4, np.minimum(e, np.abs(fz - 0.5)), e)
    seam = 1 - smoothstep(SEAM * 0.4, SEAM, e)

    # Бетон: пятна и потёки; металл: рёбра и заклёпки по краю.
    stains = fbm(px * 9, pz * 9, 11) - 0.5
    base = np.where(metal[..., None], METAL, CONCRETE) * (1 + shade[..., None] + stains[..., None] * 0.22)
    ribs = np.where(metal, 0.035 * np.sin(fx * np.pi * 12), 0.0)
    base = base * (1 + ribs[..., None])
    rx = np.abs(((fx * 8) % 1) - 0.5)
    edge = np.minimum(np.abs(fz - 0.07), np.abs(fz - 0.93))
    rivet = metal & (rx < 0.12) & (edge < 0.025)
    base = np.where(rivet[..., None], base * 1.25, base)
    col = base * (1 - seam[..., None]) + SEAM_COL * seam[..., None]

    # Несущие балки по сетке.
    n = GIRDER_EVERY
    gxs = px / PANEL  # балки идут по ровной сетке, без сдвига рядов
    dgx = np.abs(gxs - n * np.round(gxs / n))
    dgz = np.abs(gz - n * np.round(gz / n))
    dg = np.minimum(dgx, dgz)
    girder = 1 - smoothstep(GIRDER * 0.7, GIRDER, dg)
    lit = 1 - smoothstep(0.0, GIRDER * 0.35, dg)  # светлая кромка по оси балки
    gcol = GIRDER_COL * (1 + 0.35 * lit[..., None])
    return col * (1 - girder[..., None]) + gcol * girder[..., None]


def sky(d):
    """Цвет неба по направлениям d (..., 3), нормированным."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    up = np.clip(y, 0, 1)
    haze = HAZE_HORIZON + (HAZE_TOP - HAZE_HORIZON) * np.sqrt(up)[..., None]
    below = HAZE_HORIZON + (HAZE_BELOW - HAZE_HORIZON) * np.sqrt(np.clip(-y, 0, 1))[..., None]
    out = np.where((y >= 0)[..., None], haze, below)

    yy = np.maximum(y, 0.02)
    px, pz = x / yy, z / yy
    t = 1 / yy
    vis = ZENITH_VIS * np.exp(-HAZE_K * (t - 1))
    # Клочья дымки под потолком: где-то плиты видно лучше, где-то хуже.
    wisp = fbm(px * 0.5 + 3.1, pz * 0.5 - 1.7, 5)
    vis = vis * np.clip(-0.1 + 2.0 * wisp, 0, 1.6)
    vis = np.where(y > 0.02, np.clip(vis, 0, 0.75), 0.0)
    plate = ceiling(px, pz)
    lighter = (wisp - 0.5)[..., None] * 0.05
    return out * (1 - vis[..., None]) + plate * vis[..., None] + lighter * (y > 0)[..., None]


def face(name):
    c, r, dn = (np.array(v, dtype=np.float64) for v in FACES[name])
    img = np.zeros((SIZE, SIZE, 3))
    step = 128
    offs = (np.arange(SS) + 0.5) / SS
    for row in range(0, SIZE, step):
        rows = np.arange(row, min(row + step, SIZE))
        acc = np.zeros((len(rows), SIZE, 3))
        for oy in offs:
            for ox in offs:
                u = (np.arange(SIZE)[None, :] + ox) / SIZE * 2 - 1
                v = (rows[:, None] + oy) / SIZE * 2 - 1
                d = c + u[..., None] * r + v[..., None] * dn
                d = d / np.linalg.norm(d, axis=-1, keepdims=True)
                acc += sky(d)
        img[rows] = acc / (SS * SS)
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def preview(path, w=1280, h=720, pitch=35.0, fov=100.0):
    """Взгляд на север, поднятый на pitch градусов."""
    p = np.radians(pitch)
    fwd = np.array([0, np.sin(p), -np.cos(p)])
    right = np.array([1.0, 0, 0])
    upv = np.cross(right, fwd)
    k = np.tan(np.radians(fov) / 2)
    u = (np.arange(w)[None, :] + 0.5) / w * 2 - 1
    v = 1 - (np.arange(h)[:, None] + 0.5) / h * 2
    d = fwd + u[..., None] * k * right + v[..., None] * k * (h / w) * upv
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    img = sky(d)
    Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--preview-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in ([] if a.preview_only else FACES):
        face(name).save(os.path.join(OUT, name + ".png"), optimize=True)
        print("грань", name)
    if a.preview or a.preview_only:
        os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
        preview(os.path.join(ROOT, "build", "sky_preview.png"))
        print("превью build/sky_preview.png")


if __name__ == "__main__":
    main()
