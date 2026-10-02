"""Artwork for the Windows installer: wizard side image, small corner image and the app icon.

Niko Tracker - author: Nihad Jihad.
usage (any Python with Pillow, e.g. the da3 env): python installer/art/make_art.py
Writes wizard_side_*.bmp, wizard_small_*.bmp and niko.ico next to this file. Fonts: Windows'
Bahnschrift and Segoe UI (read from C:\\Windows\\Fonts, or /mnt/c/Windows/Fonts under WSL).
"""

import io
import math
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = next((p for p in (Path("C:/Windows/Fonts"), Path("/mnt/c/Windows/Fonts")) if p.exists()), None)

BG_TOP, BG_BOTTOM = (14, 17, 22), (27, 32, 41)
TEAL, CYAN, GREEN, WHITE, GREY = (47, 211, 196), (57, 198, 255), (110, 220, 120), (240, 244, 248), (140, 150, 165)


def font(name: str, size: int):
    return ImageFont.truetype(str(FONTS / name), size)


def gradient(w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h))
    px = im.load()
    for y in range(h):
        a = y / max(1, h - 1)
        c = tuple(round(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * a) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return im


def ground_grid(draw: ImageDraw.ImageDraw, w: int, h: int, horizon: float, s: float):
    """Perspective floor grid fading towards the horizon, like the add-on's guide grid."""
    vx, vy = w * 0.55, h * horizon
    for i in range(-12, 13):  # lines towards the vanishing point
        x0 = vx + i * 70 * s
        draw.line([(x0 + (x0 - vx) * 3, h * 1.6), (vx, vy)], fill=(*TEAL, 38), width=max(1, round(s)))
    for k in range(1, 14):  # depth lines, closer together far away
        t = 1 / (1 + 0.45 * k)
        y = vy + (h - vy) * t * 1.15
        if y <= vy + 2:
            continue
        alpha = round(90 * t)
        draw.line([(0, y), (w, y)], fill=(*TEAL, alpha), width=max(1, round(s)))
    # fade the floor out near the horizon
    return vy


def camera_path(draw: ImageDraw.ImageDraw, pts, s: float):
    """Solved camera path: dots shrinking with distance, a few tracked-point crosses."""
    for i, (x, y, r) in enumerate(pts):
        draw.ellipse([x - r * s, y - r * s, x + r * s, y + r * s], fill=(*CYAN, 230))
        if i % 4 == 2:
            c = 5 * s
            draw.line([(x + 14 * s - c, y - 16 * s), (x + 14 * s + c, y - 16 * s)], fill=(*GREEN, 220), width=round(1.5 * s))
            draw.line([(x + 14 * s, y - 16 * s - c), (x + 14 * s, y - 16 * s + c)], fill=(*GREEN, 220), width=round(1.5 * s))


def side_image(scale: int) -> Image.Image:
    w, h = 164 * scale, 314 * scale
    s = float(scale)
    base = gradient(w, h).convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    vy = ground_grid(d, w, h, 0.62, s)
    # camera path: an arc from the lower left up towards the horizon
    pts = []
    for i in range(16):
        t = i / 15
        x = w * (0.08 + 0.78 * t)
        y = vy + (h - vy - 44 * s) * (1 - t) ** 1.6 - 34 * s * math.sin(math.pi * t)  # clear of the credit line
        pts.append((x, y, 4.2 - 2.8 * t))
    camera_path(d, pts, s)
    glow = layer.filter(ImageFilter.GaussianBlur(3 * s))
    base = Image.alpha_composite(base, glow)
    base = Image.alpha_composite(base, layer)
    d = ImageDraw.Draw(base)
    d.text((14 * s, 22 * s), "Niko", font=font("bahnschrift.ttf", round(40 * s)), fill=WHITE)
    d.text((15 * s, 66 * s), "TRACKER", font=font("seguisb.ttf", round(15 * s)), fill=TEAL)
    d.line([(15 * s, 92 * s), (60 * s, 92 * s)], fill=TEAL, width=max(1, round(1.5 * s)))
    small = font("segoeui.ttf", round(9.5 * s))
    for k, line in enumerate(("Camera tracking", "for Blender and", "After Effects")):
        d.text((15 * s, (102 + 13 * k) * s), line, font=small, fill=GREY)
    d.text((15 * s, h - 22 * s), "by Nihad Jihad", font=font("segoeui.ttf", round(8.5 * s)), fill=GREY)
    return base.convert("RGB")


def mark(size: int) -> Image.Image:
    """The app mark: a dark rounded square, a lens ring and the camera path through it."""
    S = size * 4  # draw large, scale down for clean edges
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    r = S * 0.2
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=r, fill=(*BG_BOTTOM, 255))
    c, R = S / 2, S * 0.3
    d.ellipse([c - R, c - R, c + R, c + R], outline=(*TEAL, 255), width=round(S * 0.07))
    for i in range(7):
        t = i / 6
        x = S * (0.16 + 0.68 * t)
        y = S * (0.74 - 0.48 * t) + S * 0.08 * math.sin(math.pi * t)
        rr = S * (0.055 - 0.03 * t)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=(*CYAN, 255))
    return im.resize((size, size), Image.LANCZOS)


def small_image(scale: int) -> Image.Image:
    w = 55 * scale
    bg = Image.new("RGB", (w, w), (255, 255, 255))  # the wizard's header is white
    m = mark(round(w * 0.92))
    bg.paste(m, ((w - m.width) // 2, (w - m.height) // 2), m)
    return bg


def write_ico(path: Path, sizes=(16, 24, 32, 48, 64, 128, 256)):
    """ICO with PNG-compressed images (Windows Vista and later read them at every size)."""
    blobs = []
    for sz in sizes:
        buf = io.BytesIO()
        mark(sz).save(buf, "PNG")
        blobs.append(buf.getvalue())
    head = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = b""
    for sz, blob in zip(sizes, blobs):
        entries += struct.pack("<BBBBHHII", sz % 256, sz % 256, 0, 0, 1, 32, len(blob), offset)
        offset += len(blob)
    path.write_bytes(head + entries + b"".join(blobs))


def main():
    for k in (1, 2):
        side_image(k).save(HERE / f"wizard_side_{k}x.bmp")
        small_image(k).save(HERE / f"wizard_small_{k}x.bmp")
    write_ico(HERE / "niko.ico")
    side_image(2).save(HERE / "preview_side.png")
    mark(256).save(HERE / "preview_icon.png")
    print("art written to", HERE)


if __name__ == "__main__":
    main()
