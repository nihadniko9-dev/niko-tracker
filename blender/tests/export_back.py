"""Run after import_blender.py inside Blender: write the scene camera back as niko.blender_cameras/1.

    blender -b --factory-startup --python import_blender.py --python export_back.py -- out.json
"""

import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nikobpy  # noqa: E402

out = nikobpy.script_args()[0]
scene = bpy.context.scene
nikobpy.export_cameras(scene, scene.camera, out, scene.frame_start, scene.frame_end,
                       extra={"objects": sorted(o.name for o in scene.objects)})
