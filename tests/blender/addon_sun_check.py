"""Background check of "Add the real sun": the lamp in the levelled Blender scene must stand at the sun's
elevation and, measured from the scene's north, at its azimuth; it must follow "Set ground".

blender.exe -b --factory-startup --python tests/blender/addon_sun_check.py -- <flat solve folder with a sun> <scratch>
"""

import math
import os
import shutil
import sys

import bpy
from mathutils import Matrix, Vector

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "addon"))

import niko_tracker  # noqa: E402
from niko_tracker import ops, solve_io  # noqa: E402

src, work = (os.path.normpath(a) for a in sys.argv[sys.argv.index("--") + 1:][:2])
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(src, work)
for name in ("real_scale.json", "real_scale.json.old"):
    if os.path.exists(os.path.join(work, name)):
        os.remove(os.path.join(work, name))
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
niko_tracker.register()
ctx = bpy.context
ctx.scene.niko.solve_dir = work
s = solve_io.get(work)
sun = s.blender["sun"]
ops.build_scene(ctx, s)
assert bpy.ops.niko.add_sun() == {"FINISHED"}
ctx.view_layer.update()
lamp = bpy.data.objects[ops.SUN_NAME]
world = bpy.data.objects["Niko world"]
to_sun = (lamp.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
north = (world.matrix_world.to_3x3() @ Vector(sun["north"])).normalized()
north_h = Vector((north.x, north.y, 0)).normalized()
sun_h = Vector((to_sun.x, to_sun.y, 0)).normalized()
elev = math.degrees(math.asin(max(-1, min(1, to_sun.z))))
# azimuth: clockwise from north seen from above (+Z): east is north turned by -90 deg about Z
east_h = Vector((north_h.y, -north_h.x, 0))
az = math.degrees(math.atan2(sun_h.dot(east_h), sun_h.dot(north_h))) % 360
print(f"NIKO_SUN lamp elevation {elev:.3f} (sun {sun['elevation_deg']}), azimuth from scene north {az:.3f} "
      f"(sun {sun['azimuth_deg']}); north tilt {math.degrees(math.asin(abs(north.z))):.3f} deg")
assert abs(elev - sun["elevation_deg"]) < 0.05
assert abs((az - sun["azimuth_deg"] + 180) % 360 - 180) < 0.05
assert abs(north.z) < 1e-3  # north is level in the gimbal-levelled scene

# the lamp turns with the scene: a ground set from tilted points keeps the sun on the same scene points
before = (world.matrix_world.inverted().to_3x3() @ to_sun).normalized()
world.matrix_world = Matrix.Rotation(math.radians(10), 4, "X") @ world.matrix_world
ctx.view_layer.update()
after = (world.matrix_world.inverted().to_3x3() @ (lamp.matrix_world.to_3x3() @ Vector((0, 0, 1)))).normalized()
assert (before - after).length < 1e-6
print("NIKO_SUN OK")

# the shadow catcher lies on the floor (Z = 0) under the camera and catches shadows in Cycles
assert bpy.ops.niko.shadow_catcher() == {"FINISHED"}
ctx.view_layer.update()
pl = bpy.data.objects[ops.CATCHER_NAME]
zs = [(pl.matrix_world @ v.co).z for v in pl.data.vertices]
assert max(abs(z) for z in zs) < 1e-6 and pl.is_shadow_catcher and ctx.scene.render.film_transparent
print("NIKO_SUN shadow catcher OK, width", round(pl.dimensions.x, 3))
