"""Небо RP Sky: мир под огромным куполом из бетона и металла (референсы — Metal Garden, Blame!, Control).

Купол высоко над головой и опускается к горизонту. Он сложен из громадных плит разного размера (не сетка),
бетон в пятнах, потёках и трещинах, кое-где плиты нет — тёмный провал. Под куполом — слой облаков, у горизонта всё тонет в светлой дымке.

Каждый пиксель считается по направлению взгляда, поэтому шесть граней куба стыкуются без швов. Грани описаны
так же, как в client/SkyRenderer.java (центр, вправо, вниз по картинке) — менять только вместе.

    python scripts/gen_sky.py                 # грани в src/main/resources/assets/rpsky/textures/sky
    python scripts/gen_sky.py --preview       # ещё и превью в build/sky_preview_*.png
    python scripts/gen_sky.py --preview-only  # только превью (быстро)
"""
import argparse
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "src", "main", "resources", "assets", "rpsky", "textures", "sky")

SIZE = 1024
SS = 2  # сглаживание: SS×SS лучей на пиксель

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
# Купол: высота над головой 1, к горизонту опускается (радиус кривизны DOME_R).
DOME_R = 8.0
SLAB = 0.6  # крупная ячейка, которую потом режем на плиты (над головой ≈ 33°)
SPLITS = (0.95, 0.8, 0.6, 0.4)  # вероятность разреза на каждом уровне
SEAM = 0.007  # полуширина шва между плитами, в единицах купола
VOID_SHARE = 0.05  # плиты нет — видна решётка
METAL_SHARE = 0.18

ZENITH_VIS = 0.5  # насколько виден купол прямо над головой
HAZE_K = 0.85  # как быстро дымка густеет с расстоянием
CLOUD_H = 0.55  # высота облаков (купол — 1)
CLOUD_COVER = 0.40  # порог плотности: меньше — больше облаков

HAZE_TOP = np.array([0.56, 0.57, 0.56])
HAZE_HORIZON = np.array([0.74, 0.75, 0.75])
HAZE_BELOW = np.array([0.60, 0.61, 0.61])
CONCRETE = np.array([0.50, 0.51, 0.47])
METAL = np.array([0.36, 0.34, 0.31])
VOID = np.array([0.22, 0.23, 0.23])
SEAM_COL = np.array([0.20, 0.20, 0.19])
CLOUD_LIT = np.array([0.76, 0.77, 0.77])
CLOUD_DARK = np.array([0.48, 0.49, 0.50])

MASK = np.uint64(0xFFFFFFFF)


def hash2(ix, iz, seed):
    """Случайное 0–1 для целой точки решётки."""
    h = (np.asarray(ix).astype(np.int64).astype(np.uint64) * np.uint64(374761393)
         + np.asarray(iz).astype(np.int64).astype(np.uint64) * np.uint64(668265263)
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
    top = a + (b - a) * ux
    return top + (c + (d - c) * ux - top) * uz


def fbm(x, z, seed, octaves=4):
    s, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        s = s + value_noise(x, z, seed + o * 17) * amp
        norm += amp
        x, z, amp = x * 2.03 + 0.37, z * 2.03 - 0.71, amp * 0.5
    return s / norm


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a + (b - a) * t[..., None]


def slabs(px, pz):
    """Раскрой купола на плиты: крупная ячейка, разрезанная пополам несколько раз в случайных местах.
    Возвращает расстояние до края плиты и её номер (ix, iz, path)."""
    ix, iz = np.floor(px / SLAB), np.floor(pz / SLAB)
    x0, z0 = ix * SLAB, iz * SLAB
    x1, z1 = x0 + SLAB, z0 + SLAB
    path = np.ones_like(px, dtype=np.int64)
    for lvl, p in enumerate(SPLITS):
        key = ix + path * 7919
        do = hash2(key, iz, 100 + lvl) < p
        wide = (x1 - x0) > (z1 - z0)
        along_x = np.where(hash2(key, iz, 200 + lvl) < 0.8, wide, ~wide)
        f = 0.28 + 0.44 * hash2(key, iz, 300 + lvl)
        mid = np.where(along_x, x0 + f * (x1 - x0), z0 + f * (z1 - z0))
        side = np.where(along_x, px > mid, pz > mid)
        x0 = np.where(do & along_x & side, mid, x0)
        x1 = np.where(do & along_x & ~side, mid, x1)
        z0 = np.where(do & ~along_x & side, mid, z0)
        z1 = np.where(do & ~along_x & ~side, mid, z1)
        path = np.where(do, path * 2 + side.astype(np.int64), path)
    e = np.minimum(np.minimum(px - x0, x1 - px), np.minimum(pz - z0, z1 - pz))
    along = np.where((x1 - x0) > (z1 - z0), px, pz)  # для полос в пустых плитах
    return e, ix + path * 7919, iz, along


def dome(px, pz, detail):
    """Цвет купола в точке (px, pz). detail 0–1 гасит мелочь вдали (чтобы не рябило)."""
    e, kx, kz, along = slabs(px, pz)
    r = hash2(kx, kz, 1)
    kind = hash2(kx, kz, 2)
    void = kind < VOID_SHARE
    metal = (kind >= VOID_SHARE) & (kind < VOID_SHARE + METAL_SHARE)
    shade = (r - 0.5) * 0.26

    base = np.where(metal[..., None], METAL, CONCRETE) * (1 + shade[..., None])
    # Большие пятна и потёки, мелкая грязь.
    stains = fbm(px * 2.2, pz * 2.2, 11) - 0.5
    grime = fbm(px * 18, pz * 18, 12, 3) - 0.5
    base = base * (1 + stains[..., None] * 0.35 + grime[..., None] * 0.10 * detail[..., None])
    # Длинные царапины в двух направлениях — едва заметные (сильнее читались как тёмная штриховка).
    a = 0.35
    u1, v1 = px * np.cos(a) + pz * np.sin(a), -px * np.sin(a) + pz * np.cos(a)
    s1 = value_noise(u1 * 90, v1 * 1.3, 21) - 0.5
    s2 = value_noise(v1 * 70, u1 * 1.1, 22) - 0.5
    base = base * (1 + (s1 * 0.05 + s2 * 0.035)[..., None] * detail[..., None])
    # Трещины: у части плит — ломаная линия через плиту (прямая с изломами, не «червяк»).
    th = hash2(kx, kz, 5) * np.pi
    nx, nz = np.cos(th), np.sin(th)
    off = (hash2(kx, kz, 6) - 0.5) * 0.4
    along_c = -px * nz + pz * nx
    jag = (value_noise(along_c * 14, kx * 0.1, 33) - 0.5) * 0.03 + (value_noise(along_c * 60, kz * 0.1, 34) - 0.5) * 0.006
    d_crack = np.abs(px * nx + pz * nz - off - np.round((px * nx + pz * nz - off) / SLAB) * SLAB + jag)
    cracked = hash2(kx, kz, 7) < 0.35
    crack = np.where(cracked, 1 - smoothstep(0.002, 0.005, d_crack), 0.0) * detail
    base = mix(base, SEAM_COL, crack * 0.8)

    # Пустая плита: тёмная глубина без полос.
    base = np.where(void[..., None], VOID * (1 + stains[..., None] * 0.3), base)

    # Швы неровной ширины.
    w = SEAM * (0.7 + 0.6 * value_noise(px * 6, pz * 6, 41))
    seam = 1 - smoothstep(w * 0.45, w, e)
    col = mix(base, SEAM_COL, seam * 0.75)

    return col


def dome_hit(x, y, z):
    """Расстояние вдоль луча до купола: высота 1 над головой, опускается на s²/(2R)."""
    s2 = x * x + z * z
    a = s2 / DOME_R
    safe = np.where(a > 1e-9, a, 1.0)
    t = np.where(a > 1e-9, (-y + np.sqrt(y * y + 2 * a)) / safe, 1 / np.maximum(y, 1e-6))
    return t


def sky(d):
    """Цвет неба по направлениям d (..., 3), нормированным."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    up = np.clip(y, 0, 1)
    haze = HAZE_HORIZON + (HAZE_TOP - HAZE_HORIZON) * np.sqrt(up)[..., None]
    below = HAZE_HORIZON + (HAZE_BELOW - HAZE_HORIZON) * np.sqrt(np.clip(-y, 0, 1))[..., None]
    col = np.where((y >= 0)[..., None], haze, below)

    # Купол.
    t = dome_hit(x, y, z)
    px, pz = x * t, z * t
    vis = ZENITH_VIS * np.exp(-HAZE_K * (t - 1))
    vis = np.where(y > -0.08, vis, 0.0) * smoothstep(-0.08, 0.04, y)
    detail = np.exp(-0.9 * (t - 1))
    col = mix(col, dome(px, pz, detail), vis)

    # Облака под куполом: светлые сверху, тёмные снизу; тоже тонут в дымке.
    yc = np.maximum(y, 0.01)
    tc = CLOUD_H / yc
    cx, cz = x * tc, z * tc
    dens = fbm(cx * 1.1 + 11, cz * 1.1 - 7, 51, 5)
    cover = smoothstep(CLOUD_COVER, CLOUD_COVER + 0.2, dens)
    lit = smoothstep(-0.08, 0.12, dens - fbm(cx * 1.1 + 11.1, cz * 1.1 - 6.9, 51, 5))
    ccol = mix(CLOUD_DARK, CLOUD_LIT, lit * 0.8 + 0.2 * (1 - cover))
    cvis = np.exp(-0.55 * (tc / CLOUD_H - 1)) * smoothstep(0.0, 0.08, y)
    ccol = mix(haze, ccol, cvis)
    col = mix(col, ccol, cover * 0.9 * smoothstep(0.0, 0.05, y))
    return col


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


def preview(path, pitch, w=1280, h=720, fov=95.0):
    """Взгляд на север, поднятый на pitch градусов; нижняя треть — условная земля."""
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
    ground = d[..., 1] < -0.02
    img = np.where(ground[..., None], np.array([0.33, 0.37, 0.28]) * 0.6 + img * 0.4, img)
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
        for pitch in (12, 45, 80):
            preview(os.path.join(ROOT, "build", f"sky_preview_{pitch}.png"), pitch)
        print("превью build/sky_preview_*.png")


if __name__ == "__main__":
    main()
