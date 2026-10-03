"""Overlays: the error HUD on camera views and the error-per-frame graph on the timeline."""

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import solve_io

_handles = []
GREEN, AMBER, RED = (0.45, 0.80, 0.25, 0.9), (0.95, 0.65, 0.20, 0.9), (0.90, 0.30, 0.30, 0.9)


def _color(v, k=1.0):
    """k: clip pixels per HD pixel (solve_io.hd_scale)."""
    return GREEN if v < 0.5 * k else (AMBER if v < 1.0 * k else RED)


def _solve():
    scene = bpy.context.scene
    n = getattr(scene, "niko", None)
    return solve_io.get(n.solve_dir) if n is not None else None


def _text(x, y, size, text, rgba):
    fid = 0
    blf.size(fid, size)
    blf.enable(fid, blf.SHADOW)
    blf.shadow(fid, 5, 0.0, 0.0, 0.0, 0.85)
    blf.shadow_offset(fid, 1, -1)
    blf.color(fid, *rgba)
    blf.position(fid, x, y, 0)
    blf.draw(fid, text)
    blf.disable(fid, blf.SHADOW)
    return blf.dimensions(fid, text)


def draw_hud():
    """Camera views: the solve's average error (big, coloured) and this frame's error."""
    ctx = bpy.context
    rv3d = ctx.region_data
    if rv3d is None or rv3d.view_perspective != "CAMERA" or not ctx.scene.niko.show_hud:
        return
    s = _solve()
    if s is None:
        return
    ui = ctx.preferences.system.ui_scale
    strip = _strip(ctx, s, ui)
    x, y = 18 * ui, strip + 40 * ui  # bottom left, above the error strip (Blender's text sits top left)
    avg = s.average_px
    k = solve_io.hd_scale(s.blender.get("width"))
    label, _, rgba = solve_io.rating(avg, s.blender.get("width"),
                                     (s.report.get("solve_error") or {}).get("inlier_fraction"))
    if label == "Not reliable":
        pass
    elif s.guidance():
        label, rgba = "Needs review", (1.0, 0.65, 0.2, 1.0)
    else:
        label = "Check for sliding"
    w, _ = _text(x, y, 30 * ui, f"{avg:.2f} px" if avg is not None else "-", rgba)
    hd = f" ({avg / k:.2f} px HD)" if avg is not None and k > 1 else ""  # 4K: the rating uses HD pixels
    _text(x + w + 12 * ui, y + 4 * ui, 13 * ui, f"inlier average{hd}  |  {label}", (0.92, 0.92, 0.92, 1.0))
    f = ctx.scene.frame_current
    e = s.frame_error(f)
    # frames without a held-out measurement (between keyframes) are solved, just not scored
    line = f"frame {f}" + (f"   {e:.2f} px" if e is not None else "")
    _text(x, y - 26 * ui, 14 * ui, line, _color(e, k) if e is not None else (0.92, 0.92, 0.92, 1.0))


def _strip(ctx, s, ui) -> float:
    """Error per frame as a strip along the bottom of the camera view; returns its top (px)."""
    if not s.errors:
        return 0.0
    vals = s.errors.get("mean_px", [])
    n = len(vals)
    if not n:
        return 0.0
    region = ctx.region
    k = solve_io.hd_scale(s.blender.get("width"))
    x0, x1 = 18 * ui, region.width - 18 * ui
    y0, h = 10 * ui, 34 * ui
    scale = h / (1.5 * k)
    bw = (x1 - x0) / n
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    bg = batch_for_shader(shader, "TRIS", {"pos": [(x0, y0), (x1, y0), (x1, y0 + h), (x0, y0), (x1, y0 + h),
                                                     (x0, y0 + h)]})
    shader.uniform_float("color", (0.0, 0.0, 0.0, 0.45))
    bg.draw(shader)
    by_color = {}
    for i, v in enumerate(vals):
        if v is None:
            continue
        xa, xb = x0 + i * bw + 0.15 * bw, x0 + (i + 1) * bw - 0.15 * bw
        hh = min(h, v * scale)
        by_color.setdefault(_color(v, k), []).extend([(xa, y0), (xb, y0), (xb, y0 + hh), (xa, y0), (xb, y0 + hh),
                                                   (xa, y0 + hh)])
    for col, pts in by_color.items():
        shader.uniform_float("color", (col[0], col[1], col[2], 0.85))
        batch_for_shader(shader, "TRIS", {"pos": pts}).draw(shader)
    yl = y0 + 0.5 * k * scale
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.35))
    batch_for_shader(shader, "LINES", {"pos": [(x0, yl), (x1, yl)]}).draw(shader)
    i = ctx.scene.frame_current - s.errors.get("frame_start", 1)
    if 0 <= i < n:
        xc = x0 + (i + 0.5) * bw
        shader.uniform_float("color", (1.0, 1.0, 1.0, 0.95))
        batch_for_shader(shader, "LINES", {"pos": [(xc, y0 - 2 * ui), (xc, y0 + h + 2 * ui)]}).draw(shader)
    gpu.state.blend_set("NONE")
    return y0 + h


def draw_graph():
    """Timeline: one bar per frame, height = held-out error, dashed line at 0.5 px."""
    ctx = bpy.context
    space = ctx.space_data
    if getattr(space, "mode", None) != "TIMELINE" or not ctx.scene.niko.show_graph:
        return
    s = _solve()
    if s is None or not s.errors:
        return
    region = ctx.region
    v2d = region.view2d
    vals = s.errors.get("mean_px", [])
    f0 = s.errors.get("frame_start", 1)
    top = region.height * 0.62
    scale = top / 1.5  # 1.5 px at the top
    base = 4.0
    x0 = v2d.view_to_region(f0, 0, clip=False)[0]
    x1 = v2d.view_to_region(f0 + 1, 0, clip=False)[0]
    half = max(1.0, (x1 - x0) * 0.35)
    by_color = {}
    for i, v in enumerate(vals):
        if v is None:
            continue
        x = v2d.view_to_region(f0 + i, 0, clip=False)[0]
        if x < -10 or x > region.width + 10:
            continue
        h = min(top, v * scale)
        tris = by_color.setdefault(_color(v, solve_io.hd_scale(s.blender.get("width"))), [])
        tris += [(x - half, base), (x + half, base), (x + half, base + h),
                 (x - half, base), (x + half, base + h), (x - half, base + h)]
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    for col, pts in by_color.items():
        batch = batch_for_shader(shader, "TRIS", {"pos": pts})
        shader.uniform_float("color", (col[0], col[1], col[2], 0.55))
        batch.draw(shader)
    y = base + 0.5 * scale
    dash = []
    xx = 0.0
    while xx < region.width:
        dash += [(xx, y), (min(xx + 6, region.width), y)]
        xx += 12
    batch = batch_for_shader(shader, "LINES", {"pos": dash})
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.35))
    batch.draw(shader)
    gpu.state.blend_set("NONE")
    ui = ctx.preferences.system.ui_scale
    _text(region.width - 150 * ui, y + 3 * ui, 10 * ui, "0.5 px", (1, 1, 1, 0.6))


def register():
    _handles.append((bpy.types.SpaceView3D, bpy.types.SpaceView3D.draw_handler_add(
        draw_hud, (), "WINDOW", "POST_PIXEL")))
    _handles.append((bpy.types.SpaceDopeSheetEditor, bpy.types.SpaceDopeSheetEditor.draw_handler_add(
        draw_graph, (), "WINDOW", "POST_PIXEL")))


def unregister():
    while _handles:
        space, h = _handles.pop()
        space.draw_handler_remove(h, "WINDOW")
