"""End to end from Blender: enable the installed add-on, pick a clip, press Solve camera, wait for
the engine (WSL2) to finish, then save a screenshot of the workspace and quit.

blender.exe --factory-startup --python tests/blender/addon_gui_solve.py -- <clip> <png>
"""

import os
import sys
import time

import addon_utils
import bpy

args = sys.argv[sys.argv.index("--") + 1:]
clip, png = os.path.normpath(args[0]), os.path.normpath(args[1])
addon_utils.enable("niko_tracker", default_set=True)
_s = {"step": 0, "t0": time.time()}


def _tick():
    ctx = bpy.context
    win = ctx.window_manager.windows[0]
    n = ctx.scene.niko
    if _s["step"] == 0:
        for name in ("Cube", "Light", "Camera"):
            ob = bpy.data.objects.get(name)
            if ob is not None:
                bpy.data.objects.remove(ob, do_unlink=True)
        n.clip = clip
        n.ignore_sky = True
        with ctx.temp_override(window=win, area=win.screen.areas[0]):
            r = bpy.ops.niko.solve()
        print("NIKO_E2E solve operator:", r, flush=True)
        _s["step"] = 1
        return 2.0
    if _s["step"] == 1:
        if n.running:
            if time.time() - _s["t0"] > 1500:
                print("NIKO_E2E timeout", flush=True)
                bpy.ops.wm.quit_blender()
                return None
            return 2.0
        print("NIKO_E2E finished:", n.status, "|", n.solve_dir, flush=True)
        for st in n.stages:
            print("NIKO_E2E stage", st.label, st.state, st.detail, flush=True)
        ctx.scene.frame_set(15)
        _s["step"] = 2
        return 3.0
    if _s["step"] < 5:
        _s["step"] += 1
        for a in win.screen.areas:
            a.tag_redraw()
        return 1.0
    with ctx.temp_override(window=win, area=win.screen.areas[0]):
        bpy.ops.screen.screenshot(filepath=png)
    print("NIKO_E2E screenshot", os.path.exists(png), f"{time.time() - _s['t0']:.0f}s", flush=True)
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(_tick, first_interval=2.0)
