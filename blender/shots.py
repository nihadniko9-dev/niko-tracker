"""Benchmark shot content (run inside Blender 5.2): town scene, moving people/car proxies,
and the camera paths of the synthetic benchmark. All seeded and procedural.

Camera paths return one Blender matrix_world per frame (and optionally one lens per frame).
"""

import math
import random

import bpy
from mathutils import Euler, Matrix, Vector

import scene_lib
from scene_lib import look_at_matrix, textured_material


# --------------------------------------------------------------------------- #
# Town (drone shots)
# --------------------------------------------------------------------------- #


def build_town(seed, extent=90.0, block=18.0):
    """Ground + a grid of textured buildings of varied height, sun + sky."""
    rng = random.Random(seed)
    bpy.ops.mesh.primitive_plane_add(size=2 * extent, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "ground"
    ground.data.materials.append(textured_material("ground", rng, scale=0.12))
    n = int(extent // block)
    k = 0
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            if rng.random() < 0.18:  # open squares
                continue
            # footprints <= 10 m + 1.5 m jitter on an 18 m grid: streets >= 5 m wide at x, y = 9 + 18k
            w, d = rng.uniform(6, 10), rng.uniform(6, 10)
            h = rng.choice([rng.uniform(4, 10), rng.uniform(8, 20), rng.uniform(15, 32)])
            x = i * block + rng.uniform(-1.5, 1.5)
            y = j * block + rng.uniform(-1.5, 1.5)
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, y, h / 2))
            b = bpy.context.active_object
            b.scale = (w, d, h)
            b.rotation_euler = (0, 0, rng.choice([0.0, rng.uniform(-0.15, 0.15)]))
            b.name = f"building_{k:03d}"
            b.data.materials.append(textured_material(b.name, rng, scale=rng.uniform(0.15, 0.4)))
            k += 1
    _sun_and_sky(rng)


def _sun_and_sky(rng):
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


# --------------------------------------------------------------------------- #
# Moving people / car proxies
# --------------------------------------------------------------------------- #


def solid_material(name, rgb, roughness=0.6, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def _part(kind, name, size, loc, mat, parent, pivot_top=False, rot=(0, 0, 0)):
    if kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=size[0], depth=size[1], location=(0, 0, 0))
    elif kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=size[0], location=(0, 0, 0))
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = name
    if kind == "box":
        obj.scale = size
        bpy.ops.object.transform_apply(scale=True)
    if pivot_top:  # move the mesh so the object origin sits at its top (hip / shoulder joint)
        obj.data.transform(Matrix.Translation((0, 0, -size[1] / 2)))
    obj.rotation_euler = rot
    obj.parent = parent
    obj.location = loc
    obj.data.materials.append(mat)
    return obj


def make_person(name, rng):
    """Mannequin-like walker (~1.75 m): legs and arms swing about hip / shoulder joints."""
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    skin = solid_material(name + "_skin", rng.choice([(0.8, 0.6, 0.5), (0.55, 0.38, 0.28), (0.35, 0.24, 0.17)]), 0.5)
    shirt = solid_material(name + "_shirt", (rng.random(), rng.random(), rng.random()), 0.8)
    pants = solid_material(name + "_pants", (rng.uniform(0, 0.3), rng.uniform(0, 0.3), rng.uniform(0.1, 0.5)), 0.8)
    s = rng.uniform(0.92, 1.08)
    legs = [_part("cyl", f"{name}_leg{k}", (0.075 * s, 0.86 * s), ((k - 0.5) * 0.2 * s, 0, 0.9 * s), pants, root, True)
            for k in range(2)]
    _part("cyl", f"{name}_torso", (0.19 * s, 0.62 * s), (0, 0, 1.2 * s), shirt, root).scale = (1.0, 0.6, 1.0)
    arms = [_part("cyl", f"{name}_arm{k}", (0.05 * s, 0.62 * s), ((k - 0.5) * 0.52 * s, 0, 1.48 * s), shirt, root, True)
            for k in range(2)]
    _part("sphere", f"{name}_head", (0.11 * s,), (0, 0, 1.66 * s), skin, root)
    return root, legs, arms


def make_car(name, rng):
    """Car proxy (~4.3 x 1.8 x 1.45 m): body, cabin with dark windows, four wheels."""
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    paint = solid_material(name + "_paint", (rng.random(), rng.random(), rng.random()), 0.25, 0.6)
    glass = solid_material(name + "_glass", (0.02, 0.03, 0.04), 0.05)
    tyre = solid_material(name + "_tyre", (0.02, 0.02, 0.02), 0.9)
    _part("box", f"{name}_body", (4.3, 1.8, 0.7), (0, 0, 0.65), paint, root)
    _part("box", f"{name}_cabin", (2.2, 1.6, 0.6), (-0.3, 0, 1.3), glass, root)
    for k, (x, y) in enumerate([(1.35, 0.85), (1.35, -0.85), (-1.35, 0.85), (-1.35, -0.85)]):
        _part("cyl", f"{name}_wheel{k}", (0.34, 0.24), (x, y, 0.34), tyre, root, rot=(math.pi / 2, 0, 0))
    return root


def add_movers(cfg, n_frames, fps, frame_start):
    """People and cars crossing the area around cfg['center']. Returns [(kind, positions, headings)]."""
    if not cfg:
        return []
    rng = random.Random(cfg.get("seed", 0))
    cx, cy = cfg.get("center", (0.0, 0.0))
    radius = cfg.get("radius", 6.0)
    duration = n_frames / fps
    out = []
    # explicit movers first (e.g. a car on a street for the follow shot), then random ones
    plan = [(m["kind"], m) for m in cfg.get("explicit", [])]
    plan += [("person", None)] * cfg.get("people", 0) + [("car", None)] * cfg.get("cars", 0)
    for k, (kind, fixed) in enumerate(plan):
        if fixed is not None:
            speed = fixed["speed"]
            a = math.radians(fixed.get("heading_deg", 0.0))
            d = Vector((math.cos(a), math.sin(a), 0))
            start = Vector((*fixed["start"], 0))
        else:
            speed = rng.uniform(1.1, 1.6) if kind == "person" else rng.uniform(4.0, 8.0)
            a = rng.uniform(0, 2 * math.pi)
            mid = Vector((cx + rng.uniform(-0.5, 0.5) * radius, cy + rng.uniform(-0.5, 0.5) * radius, 0))
            d = Vector((math.cos(a), math.sin(a), 0))
            start = mid - d * speed * duration / 2
        heading = math.atan2(d.y, d.x)
        if kind == "person":
            root, legs, arms = make_person(f"person_{k:02d}", rng)
        else:
            root = make_car(f"car_{k:02d}", rng)
            legs, arms = [], []
        positions = []
        phase = rng.uniform(0, 2 * math.pi)
        for i in range(n_frames):
            t = i / fps
            p = start + d * speed * t
            f = frame_start + i
            root.location = p
            root.rotation_euler = (0, 0, heading - math.pi / 2 if kind == "person" else heading)
            root.keyframe_insert("location", frame=f)
            root.keyframe_insert("rotation_euler", frame=f)
            swing = math.sin(2 * math.pi * 0.9 * speed * t + phase)  # ~1 stride / 1.1 m
            for j, leg in enumerate(legs):
                leg.rotation_euler = (math.radians(25) * swing * (1 if j else -1), 0, 0)
                leg.keyframe_insert("rotation_euler", frame=f)
            for j, arm in enumerate(arms):
                arm.rotation_euler = (math.radians(18) * swing * (-1 if j else 1), 0, 0)
                arm.keyframe_insert("rotation_euler", frame=f)
            positions.append(tuple(p))
        out.append((kind, positions, heading))
    return out


# --------------------------------------------------------------------------- #
# Camera paths
# --------------------------------------------------------------------------- #


def smooth_noise(n, fps, amp, rng, fmin=0.3, fmax=6.0, terms=6):
    """Sum of sines with 1/f amplitudes, RMS ~ amp. Deterministic for a seeded rng."""
    freqs = [fmin * (fmax / fmin) ** (k / (terms - 1)) for k in range(terms)]
    weights = [1.0 / f for f in freqs]
    norm = math.sqrt(sum(w * w for w in weights) / 2)
    phases = [rng.uniform(0, 2 * math.pi) for _ in freqs]
    return [amp / norm * sum(w * math.sin(2 * math.pi * f * i / fps + p) for f, w, p in zip(freqs, weights, phases))
            for i in range(n)]


def _cam(eye, yaw_deg, pitch_deg, roll_deg=0.0):
    """Camera at eye looking along yaw (from +X toward +Y) and pitch (up positive)."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = Vector((math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)))
    return look_at_matrix(eye, Vector(eye) + fwd, roll_deg)


def _shake(M, dyaw, dpitch, droll):
    """Apply small rotations in the camera's own frame (pitch about X, yaw about Y, roll about Z)."""
    R = M.to_3x3() @ Euler((math.radians(dpitch), math.radians(dyaw), math.radians(droll)), "XYZ").to_matrix()
    return Matrix.Translation(M.translation) @ R.to_4x4()


def camera_path(cfg, n, fps, movers):
    """Dispatch on cfg['path']. Returns (matrices, lenses or None)."""
    rng = random.Random(cfg.get("seed", 1))
    kind = cfg["path"]
    lens0 = cfg.get("lens", 28.0)
    lenses = None
    ease = [0.5 - 0.5 * math.cos(math.pi * i / max(n - 1, 1)) for i in range(n)]
    shake = cfg.get("shake_deg", 0.0)
    jy = smooth_noise(n, fps, shake, rng)
    jp = smooth_noise(n, fps, shake * 0.8, rng)
    jr = smooth_noise(n, fps, shake * 0.5, rng)
    mats = []

    if kind == "orbit":
        mats = scene_lib.orbit_path(n, cfg.get("target", (0, 0, 1)), cfg["radius"], cfg.get("height", 2.0),
                                    cfg.get("arc_deg", 40.0), cfg.get("start_deg", 0.0))
    elif kind == "walk":  # handheld walk: forward motion, step bob, sway, hand shake
        start = Vector(cfg["start"])
        yaw = cfg.get("yaw_deg", 0.0)
        d = Vector((math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0))
        side = Vector((-d.y, d.x, 0))
        speed = cfg.get("speed", 1.2)
        step_hz = cfg.get("step_hz", 1.8)
        for i in range(n):
            t = i / fps
            eye = start + d * speed * t
            eye.z += 0.025 * math.sin(2 * math.pi * step_hz * t)
            eye += side * 0.015 * math.sin(math.pi * step_hz * t)
            yaw_i = yaw + cfg.get("turn_deg", 0.0) * ease[i]
            mats.append(_shake(_cam(eye, yaw_i, cfg.get("pitch_deg", -5.0)), jy[i], jp[i], jr[i]))
    elif kind == "line":  # dolly / drone fly-over: straight line, optional yaw drift
        a, b = Vector(cfg["start"]), Vector(cfg["end"])
        for i in range(n):
            s = i / max(n - 1, 1)
            eye = a.lerp(b, s)
            yaw_i = cfg.get("yaw_deg", 0.0) + cfg.get("yaw_drift_deg", 0.0) * (s - 0.5)
            mats.append(_shake(_cam(eye, yaw_i, cfg.get("pitch_deg", 0.0)), jy[i], jp[i], jr[i]))
    elif kind == "pan":  # tripod-ish: fixed point (+ small nodal offset), yaw sweep, optional tilt
        eye = Vector(cfg["eye"])
        offset = cfg.get("nodal_offset_m", 0.0)  # camera centre circles the pan axis -> tiny parallax
        for i in range(n):
            yaw_i = cfg.get("yaw_start_deg", 0.0) + cfg.get("yaw_sweep_deg", 30.0) * ease[i]
            pitch_i = cfg.get("pitch_deg", 0.0) + cfg.get("tilt_sweep_deg", 0.0) * math.sin(math.pi * ease[i])
            yr = math.radians(yaw_i)
            e = eye + Vector((math.cos(yr), math.sin(yr), 0)) * offset
            mats.append(_shake(_cam(e, yaw_i, pitch_i), jy[i], jp[i], jr[i]))
    elif kind == "whip":  # slow pan, a fast whip in `whip_frames`, slow pan again
        eye = Vector(cfg["eye"])
        w0, wn = cfg.get("whip_start", n // 2 - 5), cfg.get("whip_frames", 10)
        yaw = cfg.get("yaw_start_deg", 0.0)
        for i in range(n):
            if i < w0:
                y_i = yaw + cfg.get("slow_deg_per_frame", 0.1) * i
            elif i < w0 + wn:
                s = (i - w0) / wn
                y_i = yaw + cfg.get("slow_deg_per_frame", 0.1) * w0 + cfg["whip_deg"] * (0.5 - 0.5 * math.cos(math.pi * s))
            else:
                y_i = (yaw + cfg.get("slow_deg_per_frame", 0.1) * (i - wn) + cfg["whip_deg"])
            mats.append(_shake(_cam(eye, y_i, cfg.get("pitch_deg", 0.0)), jy[i], jp[i], jr[i]))
    elif kind == "follow":  # chase the first car mover from behind
        cars = [m for m in movers if m[0] == "car"]
        if not cars:
            raise ValueError("follow path needs a car mover")
        _, pos, heading = cars[0]
        back = Vector((-math.cos(heading), -math.sin(heading), 0))
        for i in range(n):
            p = Vector(pos[i])
            eye = p + back * cfg.get("distance", 9.0) + Vector((0, 0, cfg.get("height", 2.2)))
            M = look_at_matrix(eye, p - back * 4.0 + Vector((0, 0, 0.8)))
            mats.append(_shake(M, jy[i], jp[i], jr[i]))
    else:
        raise ValueError(f"unknown camera path {kind!r}")

    if "lens_end" in cfg:  # zoom
        lenses = [lens0 + (cfg["lens_end"] - lens0) * e for e in ease]
    return mats, lenses
