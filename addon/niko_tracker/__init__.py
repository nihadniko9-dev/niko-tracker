"""Niko Tracker - camera tracking for Blender, driven by the Niko Tracker Engine (WSL2).

Niko Tracker Engine - author: Nihad Jihad ("Niko").
Copyright (c) 2026 Nihad Jihad ("Niko"). All rights reserved; see LICENSE.
Adds a "Niko track" workspace and a "Niko" tab in the 3D view sidebar:
1 clip, 2 ignore, 3 solve, 4 result, 5 use it (camera, lock view, lock test MP4, After Effects).
The engine runs in WSL2 (niko solve); Blender only reads its files and does no camera maths.
"""

bl_info = {
    "name": "Niko Tracker",
    "author": "Nihad Jihad (Niko)",
    "version": (0, 3, 8),
    "blender": (5, 2, 0),
    "location": "Workspace tab 'Niko track' / 3D View > Sidebar > Niko",
    "description": "Automatic camera tracking with the Niko Tracker Engine",
    "category": "Motion Tracking",
}

from . import draw, ops, props, ui  # noqa: E402

_modules = (props, ops, ui, draw)


def register():
    for m in _modules:
        m.register()


def unregister():
    for m in reversed(_modules):
        m.unregister()
