"""GUI check of camera projection: scene mesh imported, footage projected through the solved camera, screenshot.

blender.exe --factory-startup --python tests/blender/addon_gui_screenshot.py -- <solve folder> <png> [frame]
Blender opens a window, builds the workspace, waits for a few redraws, saves the screenshot, quits.
"""

import os
import sys

import bpy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
folder, png = os.path.normpath(args[0]), os.path.normpath(args[1])
frame = int(args[2]) if len(args) > 2 else None
niko_tracker.register()
_state = {"step": 0}


def _tick():
    ctx = bpy.context
    step = _state["step"]
    _state["step"] += 1
    if step == 0:
        for name in ("Cube", "Light", "Camera"):  # factory-startup objects, not part of the check
            ob = bpy.data.objects.get(name)
            if ob is not None:
                bpy.data.objects.remove(ob, do_unlink=True)
        ctx.scene.niko.solve_dir = folder
        ctx.scene.niko.status = "Solved: example solve loaded for the screenshot"
        ops.build_scene(ctx, solve_io.get(folder))
        mp = os.path.join(solve_io.selected_dir(folder), "mesh.ply")
        if os.path.exists(mp):
            win0 = ctx.window_manager.windows[0]
            with ctx.temp_override(window=win0, screen=win0.screen):
                ops.import_scene_mesh(bpy.context, mp)
                ops.project_footage(bpy.context, True)
        win = ctx.window_manager.windows[0]
        with ctx.temp_override(window=win):
            ops.setup_workspace(bpy.context)
        if frame is not None:
            ctx.scene.frame_set(frame)
        return 1.0
    if step < 20:
        for win in ctx.window_manager.windows:
            for area in win.screen.areas:
                area.tag_redraw()
        return 1.0
    win = ctx.window_manager.windows[0]
    with ctx.temp_override(window=win, area=win.screen.areas[0]):
        bpy.ops.screen.screenshot(filepath=png)
    print("NIKO_SCREENSHOT", png, os.path.exists(png))
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(_tick, first_interval=2.0)
