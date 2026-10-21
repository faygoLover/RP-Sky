"""Небо RP Sky: мир под огромным куполом из бетона и металла (референсы — Metal Garden, Blame!, Control).

Купол высоко над головой, почти плоский (мир плоский), чуть опускается к горизонту. Он сложен из бетонных панелей рядами
(каждый второй ряд сдвинут на полпанели, изредка панель поделена), швы тонкие и местами не видны; бетон с зерном,
сырыми пятнами, потёками сверху вниз, высолами, раковинами, редкими трещинами и сколами. Под куполом — слой облаков, у горизонта всё тонет в светлой дымке.

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

SIZE = 2048
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
# Купол: высота над головой 1, к горизонту чуть опускается (радиус кривизны DOME_R; больше — площе).
DOME_R = 40.0
PANEL_W = 0.30  # панель: ширина и высота на потолке высотой 1 (над головой ≈ 17° × 25°)
PANEL_H = 0.45
SPLIT_SHARE = 0.12  # доля панелей, поделённых пополам
SEAM = 0.0025  # полуширина шва

ZENITH_VIS = 0.68  # насколько виден купол прямо над головой
HAZE_K = 0.75  # как быстро дымка густеет с расстоянием
CLOUD_H = 0.55  # высота облаков (купол — 1)
CLOUD_COVER = 0.47  # порог плотности: меньше — больше облаков

HAZE_TOP = np.array([0.56, 0.57, 0.56])
HAZE_HORIZON = np.array([0.74, 0.75, 0.75])
HAZE_BELOW = np.array([0.60, 0.61, 0.61])
CONCRETE = np.array([0.53, 0.52, 0.49])
STAIN = np.array([0.33, 0.33, 0.29])
SEAM_COL = np.array([0.24, 0.24, 0.23])
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
    """Раскладка панелей: ряды одинаковых прямоугольных панелей, каждый второй ряд сдвинут на полпанели;
    изредка панель поделена пополам. Возвращает расстояние до края, номер панели (kx, kz) и её границы."""
    iz = np.floor(pz / PANEL_H)
    shift = np.where(iz % 2 == 0, 0.0, PANEL_W / 2)
    ix = np.floor((px - shift) / PANEL_W)
    x0, z0 = ix * PANEL_W + shift, iz * PANEL_H
    x1, z1 = x0 + PANEL_W, z0 + PANEL_H
    split = hash2(ix, iz, 100) < SPLIT_SHARE
    mid = (z0 + z1) / 2
    lower = pz > mid
    z0 = np.where(split & lower, mid, z0)
    z1 = np.where(split & ~lower, mid, z1)
    kz = iz * 2 + np.where(split & lower, 1, 0)
    e = np.minimum(np.minimum(px - x0, x1 - px), np.minimum(pz - z0, z1 - pz))
    return e, ix, kz, x0, x1, z0, z1


def dome(px, pz, detail):
    """Цвет купола в точке (px, pz). detail 0–1 гасит мелочь вдали (чтобы не рябило)."""
    e, kx, kz, x0, x1, z0, z1 = slabs(px, pz)
    dt = detail[..., None]

    # Один бетон; панели чуть различаются тоном.
    tone = 1 + (hash2(kx, kz, 1) - 0.5) * 0.07
    base = CONCRETE * tone[..., None]

    # Зерно, крапинки и мелкая пестрота.
    grain = (value_noise(px * 420, pz * 420, 61) - 0.5) * 0.10 + (value_noise(px * 140, pz * 140, 62) - 0.5) * 0.08
    speck = smoothstep(0.80, 0.92, value_noise(px * 260, pz * 260, 65)) - smoothstep(0.84, 0.95, value_noise(px * 230, pz * 230, 66))
    mottle = fbm(px * 6, pz * 6, 63, 5) - 0.5
    base = base * (1 + grain[..., None] * dt + speck[..., None] * 0.10 * dt + mottle[..., None] * 0.14)

    # Сырые пятна с рваными краями: крупные, тёмные, буро-зелёные.
    s = fbm(px * 1.5, pz * 1.5, 11, 6) + (fbm(px * 25, pz * 25, 64, 3) - 0.5) * 0.10
    blot = smoothstep(0.52, 0.60, s)
    inner = smoothstep(0.58, 0.70, s)
    base = mix(base, STAIN, blot * 0.7 + inner * 0.3)

    # Потёки сверху вниз по панели (от верхнего шва), неровные, разной длины.
    h = np.maximum(z1 - z0, 1e-6)
    dz = (pz - z0) / h
    lane = value_noise(px * 70, kz * 0.37, 71)  # «дорожки» потёков поперёк панели
    length = 0.15 + 0.75 * value_noise(px * 18, kz * 0.53, 72)
    drip = smoothstep(0.55, 0.75, lane) * (1 - smoothstep(length * 0.6, length, dz))
    drip = drip * (0.5 + 0.5 * value_noise(px * 300, pz * 40, 73))
    base = mix(base, STAIN, drip * 0.45 * detail)

    # Белёсые высолы — светлые разводы.
    salt = smoothstep(0.62, 0.72, fbm(px * 3 + 7, pz * 3 - 2, 67, 5))
    base = base * (1 + 0.10 * salt[..., None])

    # Раковины — мелкие тёмные точки.
    pit = smoothstep(0.92, 0.97, value_noise(px * 200, pz * 200, 91)) * detail
    base = base * (1 - 0.4 * pit[..., None])

    # Трещины — редкие, тонкие, ломаные.
    th = hash2(kx, kz, 5) * np.pi
    nx, nz = np.cos(th), np.sin(th)
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    off = (hash2(kx, kz, 6) - 0.5) * 0.5 * np.minimum(x1 - x0, z1 - z0)
    along_c = -(px - cx) * nz + (pz - cz) * nx
    jag = (value_noise(along_c * 16, kx * 0.1, 33) - 0.5) * 0.02 + (value_noise(along_c * 70, kz * 0.1, 34) - 0.5) * 0.004
    d_crack = np.abs((px - cx) * nx + (pz - cz) * nz - off + jag)
    cracked = hash2(kx, kz, 7) < 0.12
    crack = np.where(cracked, 1 - smoothstep(0.001, 0.0028, d_crack), 0.0) * detail
    base = mix(base, SEAM_COL, crack * 0.6)

    # Сколы у шва — редко.
    cn = fbm(px * 22, pz * 22, 81, 3)
    chip = smoothstep(0.72, 0.76, cn) * (1 - smoothstep(0.004, 0.0042 + 0.04 * np.clip(cn - 0.7, 0, 1), e))
    base = mix(base, base * 0.7, chip * 0.8)

    # Швы тонкие, местами почти не видны.
    seen = 0.15 + 0.6 * smoothstep(0.35, 0.7, value_noise(px * 4, pz * 4, 41))
    seam = 1 - smoothstep(SEAM * 0.4, SEAM, e)
    return mix(base, SEAM_COL, seam * seen)

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
