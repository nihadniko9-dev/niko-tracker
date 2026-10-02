"""Procedural, well-textured scene building blocks (run inside Blender 5.2).

Everything is procedural (no external assets), seeded, and dense in texture at
many scales so point trackers have features at any distance.
"""

import math
import random

import bpy
from mathutils import Matrix, Vector


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #


CONTRAST = 1.0  # < 1 flattens every procedural texture (low-texture shots); set by gen_shot


def _c(v):
    """Pull a 0..1 value toward mid-grey by CONTRAST (no extra random draws: shots stay reproducible)."""
    return 0.5 + (v - 0.5) * CONTRAST


def _ramp(nodes, rng, n=4):
    ramp = nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    while len(els) < n:
        els.new(0.5)
    for k, el in enumerate(els):
        el.position = k / (n - 1)
        v = rng.uniform(0.05, 0.95)
        hue_shift = rng.uniform(-0.25, 0.25)
        rgb = (min(1, max(0, v + hue_shift)), v, min(1, max(0, v - hue_shift)))
        el.color = (*(_c(x) for x in rgb), 1.0)
    return ramp


def textured_material(name, rng, scale=1.0):
    """Principled BSDF whose base colour is a random multi-scale procedural pattern."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.55, 0.95)

    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    mapping.inputs["Rotation"].default_value = (rng.uniform(0, 6.3), rng.uniform(0, 6.3), rng.uniform(0, 6.3))
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    vec = mapping.outputs["Vector"]

    # Two detail layers present in every recipe, pushed to high contrast so trackers
    # find corners at close and middle range: fine noise and mid-scale Voronoi cells.
    fine = nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = rng.uniform(30, 70)
    fine.inputs["Detail"].default_value = 8.0
    fine.inputs["Roughness"].default_value = 0.7
    links.new(vec, fine.inputs["Vector"])
    fine_ramp = nodes.new("ShaderNodeValToRGB")
    fine_ramp.color_ramp.elements[0].position = 0.38
    fine_ramp.color_ramp.elements[1].position = 0.62
    fine_ramp.color_ramp.elements[0].color = (*[_c(0.0)] * 3, 1.0)
    fine_ramp.color_ramp.elements[1].color = (*[_c(1.0)] * 3, 1.0)
    links.new(fine.outputs["Fac"], fine_ramp.inputs["Fac"])

    cells = nodes.new("ShaderNodeTexVoronoi")
    cells.inputs["Scale"].default_value = rng.uniform(8, 20)
    links.new(vec, cells.inputs["Vector"])
    cells_ramp = _ramp(nodes, rng, 3)
    links.new(cells.outputs["Distance"], cells_ramp.inputs["Fac"])

    recipe = rng.choice(["brick", "checker", "voronoi", "magic", "noise"])
    if recipe == "brick":
        tex = nodes.new("ShaderNodeTexBrick")
        tex.inputs["Scale"].default_value = rng.uniform(2, 8)
        tex.inputs["Mortar Size"].default_value = rng.uniform(0.01, 0.04)
        tex.inputs["Color1"].default_value = (*[_c(rng.uniform(0.2, 0.9))] * 3, 1)
        tex.inputs["Color2"].default_value = (*[_c(rng.uniform(0.1, 0.6))] * 3, 1)
        links.new(vec, tex.inputs["Vector"])
        base = tex.outputs["Color"]
    elif recipe == "checker":
        tex = nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = rng.uniform(2, 10)
        links.new(vec, tex.inputs["Vector"])
        ramp = _ramp(nodes, rng, 2)
        links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        base = ramp.outputs["Color"]
    elif recipe == "voronoi":
        tex = nodes.new("ShaderNodeTexVoronoi")
        tex.inputs["Scale"].default_value = rng.uniform(3, 15)
        links.new(vec, tex.inputs["Vector"])
        ramp = _ramp(nodes, rng, 4)
        links.new(tex.outputs["Distance"], ramp.inputs["Fac"])
        base = ramp.outputs["Color"]
    elif recipe == "magic":
        tex = nodes.new("ShaderNodeTexMagic")
        tex.inputs["Scale"].default_value = rng.uniform(8, 16)
        tex.turbulence_depth = 3
        links.new(vec, tex.inputs["Vector"])
        base = tex.outputs["Color"]
    else:
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = rng.uniform(3, 10)
        tex.inputs["Detail"].default_value = 12.0
        tex.inputs["Roughness"].default_value = 0.65
        links.new(vec, tex.inputs["Vector"])
        ramp = _ramp(nodes, rng, 5)
        links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        base = ramp.outputs["Color"]

    mid = nodes.new("ShaderNodeMix")
    mid.data_type = "RGBA"
    mid.blend_type = "OVERLAY"
    mid.inputs["Factor"].default_value = rng.uniform(0.4, 0.7) * min(1.0, CONTRAST)
    links.new(base, mid.inputs["A"])
    links.new(cells_ramp.outputs["Color"], mid.inputs["B"])

    top = nodes.new("ShaderNodeMix")
    top.data_type = "RGBA"
    top.blend_type = "OVERLAY"
    top.inputs["Factor"].default_value = rng.uniform(0.45, 0.75) * min(1.0, CONTRAST)
    links.new(mid.outputs["Result"], top.inputs["A"])
    links.new(fine_ramp.outputs["Color"], top.inputs["B"])
    links.new(top.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _add(kind, location, size, rotation_z=0.0):
    if kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
        obj = bpy.context.active_object
        obj.scale = size
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.5, depth=1.0, location=location)
        obj = bpy.context.active_object
        obj.scale = size
    else:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=0.5, location=location)
        obj = bpy.context.active_object
        obj.scale = size
    obj.rotation_euler = (0.0, 0.0, rotation_z)
    return obj


def _near_segments(x, y, segments):
    """True if (x, y) is within the half-width of any (x0, y0, x1, y1, half_width) segment."""
    for x0, y0, x1, y1, hw in segments:
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        s = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
        if math.hypot(x - (x0 + s * dx), y - (y0 + s * dy)) < hw:
            return True
    return False


def build_yard(seed, extent=20.0, n_objects=30, clear_radius=0.0, clear_segments=(), clear_width=2.0):
    """Textured ground, scattered props, perimeter walls, sun + sky. Returns the static objects.

    clear_radius keeps an annulus of +-clear_width free (orbits); clear_segments keeps corridors
    free (walks)."""
    rng = random.Random(seed)
    objs = []

    bpy.ops.mesh.primitive_plane_add(size=2 * extent, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(textured_material("ground", rng, scale=0.35))
    objs.append(ground)

    placed = 0
    tries = 0
    while placed < n_objects and tries < 20 * n_objects:
        tries += 1
        r = math.sqrt(rng.uniform(0, 1)) * (extent * 0.8)
        a = rng.uniform(0, 2 * math.pi)
        x, y = r * math.cos(a), r * math.sin(a)
        if clear_radius and abs(r - clear_radius) < clear_width:
            continue  # keep the camera path clear
        if clear_segments and _near_segments(x, y, clear_segments):
            continue
        kind = rng.choice(["box", "box", "cylinder", "sphere"])
        s = (rng.uniform(0.4, 3.0), rng.uniform(0.4, 3.0), rng.uniform(0.4, 3.5))
        if kind != "box":
            s = (s[0], s[0], s[2])
        obj = _add(kind, (x, y, s[2] / 2), s, rng.uniform(0, math.pi))
        obj.name = f"prop_{placed:03d}"
        obj.data.materials.append(textured_material(obj.name, rng, scale=rng.uniform(0.3, 1.2)))
        objs.append(obj)
        placed += 1

    for k in range(4):  # perimeter walls: texture in the background
        a = k * math.pi / 2
        loc = (extent * math.cos(a), extent * math.sin(a), 3.0)
        obj = _add("box", loc, (0.5, 2 * extent, 6.0), a)
        obj.name = f"wall_{k}"
        obj.data.materials.append(textured_material(obj.name, rng, scale=0.25))
        objs.append(obj)

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.0
    sun.angle = math.radians(2.0)
    sun_obj = bpy.data.objects.new("sun", sun)
    sun_obj.rotation_euler = (math.radians(rng.uniform(30, 55)), 0, math.radians(rng.uniform(0, 360)))
    bpy.context.scene.collection.objects.link(sun_obj)

    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.45, 0.6, 0.85, 1.0)
    bg.inputs["Strength"].default_value = 0.8
    bpy.context.scene.world = world
    return objs


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def look_at_matrix(eye, target, roll_deg=0.0):
    """Blender camera matrix_world at `eye` looking at `target` (camera -Z forward, +Y up)."""
    eye, target = Vector(eye), Vector(target)
    forward = (target - eye).normalized()
    rot = forward.to_track_quat("-Z", "Y").to_matrix()
    if roll_deg:
        rot = rot @ Matrix.Rotation(math.radians(roll_deg), 3, "Z")
    return Matrix.Translation(eye) @ rot.to_4x4()


def orbit_path(n, target, radius, height, arc_deg, start_deg=0.0):
    out = []
    for i in range(n):
        a = math.radians(start_deg + arc_deg * i / max(n - 1, 1))
        eye = (target[0] + radius * math.cos(a), target[1] + radius * math.sin(a), height)
        out.append(look_at_matrix(eye, target))
    return out


def keyframe_camera(cam_obj, matrices, frame_start, lenses=None):
    """One key per frame; linear interpolation so sub-frame (motion blur, rolling shutter)
    poses stay on the path. Quaternion signs are kept continuous (q and -q are the same
    rotation, but linear interpolation between them is not)."""
    cam_obj.rotation_mode = "QUATERNION"
    prev = None
    for i, M in enumerate(matrices):
        cam_obj.matrix_world = M
        q = cam_obj.rotation_quaternion.copy()
        if prev is not None and q.dot(prev) < 0:
            q.negate()
            cam_obj.rotation_quaternion = q
        prev = q
        f = frame_start + i
        cam_obj.keyframe_insert("location", frame=f)
        cam_obj.keyframe_insert("rotation_quaternion", frame=f)
        if lenses is not None:
            cam_obj.data.lens = lenses[i]
            cam_obj.data.keyframe_insert("lens", frame=f)
    for idblock in (cam_obj, cam_obj.data):
        anim = idblock.animation_data
        if anim and anim.action:
            for fc in _fcurves(anim):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


def _fcurves(anim):
    """F-curves of an object's action (Blender 5.x layered actions or legacy)."""
    action = anim.action
    if hasattr(action, "fcurves") and len(getattr(action, "fcurves", [])):
        return list(action.fcurves)
    out = []
    slot = getattr(anim, "action_slot", None)
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            bag = strip.channelbag(slot) if slot is not None else None
            if bag is not None:
                out.extend(bag.fcurves)
    return out


def sample_scene_points(scene, cam_obj, frames, grid=(16, 9), max_dist=1e4):
    """Ray-cast a pixel grid per frame. Returns hit points (world) and per-frame hit distances."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hits, dists = [], []
    for f in frames:
        scene.frame_set(f)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mw = cam_obj.matrix_world.normalized()
        tr, br, bl, tl = cam_obj.data.view_frame(scene=scene)
        origin = mw.translation
        for gy in range(grid[1]):
            for gx in range(grid[0]):
                u = (gx + 0.5) / grid[0]
                v = (gy + 0.5) / grid[1]
                top = tl.lerp(tr, u)
                bottom = bl.lerp(br, u)
                local = top.lerp(bottom, v)
                direction = (mw.to_3x3() @ local).normalized()
                ok, loc, _n, _i, _o, _m = scene.ray_cast(depsgraph, origin, direction, distance=max_dist)
                if ok:
                    hits.append(tuple(loc))
                    dists.append((loc - origin).length)
    return hits, dists
