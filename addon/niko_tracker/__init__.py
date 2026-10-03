"""Niko Tracker - camera tracking for Blender, driven by the Niko Tracker Engine (WSL2).

Niko Tracker Engine - author: Nihad Jiad Yousef.
Copyright (c) 2026 Nihad Jiad Yousef. All rights reserved; see LICENSE.
Adds a "Niko track" workspace and a "Niko" tab in the 3D view sidebar:
1 choose video, 2 solve camera, 3 check result; advanced lens/mask controls and camera export.
The engine runs in WSL2 (niko solve); Blender only reads its files and does no camera maths.
"""

bl_info = {
    "name": "Niko Tracker",
    "author": "Nihad Jiad Yousef",
    "version": (0, 5, 0),
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
