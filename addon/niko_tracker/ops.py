"""Operators: solve (engine job with live progress), load a solve, build the scene, workspace, exports."""

import datetime
import math
import os
import re
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import engine, solve_io

COLLECTION = "Niko track"
WORKSPACE = "Niko track"
MAX_POINTS = 6000
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mxf", ".avi", ".mkv", ".webm", ".mts", ".m2ts", ".mpg", ".mpeg"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".dpx", ".bmp"}


def prefs(context):
    return context.preferences.addons[__package__].preferences


def _redraw(context):
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            area.tag_redraw()


_HOME_CACHE = {}


def niko_home_windows(distro: str) -> tuple[str, str]:
    """($NIKO_HOME inside WSL, the same folder as Windows sees it)."""
    if distro not in _HOME_CACHE:
        home = engine.query(distro, 'printf %s "$NIKO_HOME"')
        if not home:
            raise RuntimeError(f"WSL distribution '{distro}' has no NIKO_HOME: is the engine installed?")
        _HOME_CACHE[distro] = (home, engine.to_windows(home, distro))
    return _HOME_CACHE[distro]


# --------------------------------------------------------------------------------------------- scene

def _clear_collection(coll):
    for ob in list(coll.objects):
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Camera):
                bpy.data.cameras.remove(data)
            elif isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)


def _points_modifier(ob, radius: float):
    """Show a vertex-only mesh as small spheres (Geometry Nodes 'Mesh to Points')."""
    ng = bpy.data.node_groups.get("Niko points")
    if ng is None:
        ng = bpy.data.node_groups.new("Niko points", "GeometryNodeTree")
        ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        gi, go = ng.nodes.new("NodeGroupInput"), ng.nodes.new("NodeGroupOutput")
        m2p = ng.nodes.new("GeometryNodeMeshToPoints")
        setm = ng.nodes.new("GeometryNodeSetMaterial")
        gi.location, m2p.location, setm.location, go.location = (-400, 0), (-150, 0), (100, 0), (350, 0)
        ng.links.new(gi.outputs[0], m2p.inputs["Mesh"])
        ng.links.new(m2p.outputs["Points"], setm.inputs["Geometry"])
        ng.links.new(setm.outputs["Geometry"], go.inputs[0])
        mat = bpy.data.materials.get("Niko points") or bpy.data.materials.new("Niko points")
        mat.diffuse_color = (0.15, 0.85, 1.0, 1.0)
        setm.inputs["Material"].default_value = mat
    mod = ob.modifiers.new("Niko points", "NODES")
    mod.node_group = ng
    mod.show_in_editmode = False  # in Edit Mode the points themselves are shown and picked
    for node in ng.nodes:
        if node.bl_idname == "GeometryNodeMeshToPoints":
            node.inputs["Radius"].default_value = radius


def build_scene(context, s: solve_io.Solve):
    """Camera (one key per frame, values straight from the engine), footage behind it, points,
    all under an empty 'Niko world' that levels the scene (ground on Z = 0)."""
    d = s.blender
    scene = context.scene
    r = scene.render
    r.resolution_x, r.resolution_y, r.resolution_percentage = d["width"], d["height"], 100
    r.pixel_aspect_x, r.pixel_aspect_y = d["pixel_aspect_x"], d["pixel_aspect_y"]
    r.fps, r.fps_base = d["fps_int"], d["fps_int"] / d["fps"]
    scene.frame_start = d["frame_start"]
    scene.frame_end = d["frame_start"] + len(d["frames"]) - 1

    coll = bpy.data.collections.get(COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(COLLECTION)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    _clear_collection(coll)

    world = bpy.data.objects.new("Niko world", None)
    world.empty_display_type = "PLAIN_AXES"
    adjust = solve_io.user_adjust(s.folder)  # size / ground set in Blender
    world.matrix_world = (Matrix(adjust) if adjust else Matrix.Identity(4)) @ _engine_world(d)
    coll.objects.link(world)

    cam_data = bpy.data.cameras.new("Niko camera")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = d["sensor_width"]
    cam = bpy.data.objects.new("Niko camera", cam_data)
    coll.objects.link(cam)
    cam.parent = world
    cam.matrix_parent_inverse = Matrix.Identity(4)
    cam.rotation_mode = "QUATERNION"
    prev = None
    for f in d["frames"]:
        if not f["valid"]:
            continue
        cam.matrix_basis = Matrix(f["matrix_world"])  # engine values, in the solve's own world
        q = cam.rotation_quaternion.copy()
        if prev is not None and q.dot(prev) < 0:  # same rotation, continuous sign for sub-frame motion blur
            q.negate()
            cam.rotation_quaternion = q
        prev = q
        cam.keyframe_insert("location", frame=f["frame"])
        cam.keyframe_insert("rotation_quaternion", frame=f["frame"])
        cam_data.lens = f["lens"]
        cam_data.shift_x, cam_data.shift_y = f["shift_x"], f["shift_y"]
        for prop in ("lens", "shift_x", "shift_y"):
            cam_data.keyframe_insert(prop, frame=f["frame"])
    scene.camera = cam
    _fit_camera_to_scale(cam_data, world, d)
    cam_data.passepartout_alpha = 0.85

    if not context.scene.niko.clip and s.source_clip():  # a loaded solve brings its own footage
        context.scene.niko.clip = s.source_clip()
    _footage(context, cam_data, d)

    if len(s.points):
        pts_xyz = s.points
        if len(pts_xyz) > MAX_POINTS:  # a readable cloud; every point stays in points.ply
            pts_xyz = pts_xyz[np.random.default_rng(0).choice(len(pts_xyz), MAX_POINTS, replace=False)]
        me = bpy.data.meshes.new("Niko points")
        me.vertices.add(len(pts_xyz))
        me.vertices.foreach_set("co", pts_xyz.ravel())
        me.update()
        pts = bpy.data.objects.new("Niko points", me)
        coll.objects.link(pts)
        pts.parent = world
        pts.matrix_parent_inverse = Matrix.Identity(4)
        scale = max(world.matrix_world.to_scale())
        # ~2.5 px on screen at the median depth whatever the lens: a fixed size covered the footage
        # with a telephoto lens (real clip 03 at 80 mm). Depth in solve units (blender.json from
        # 0.4 on; before, the levelled scene put the median depth at 10 units)
        lens = [f["lens"] for f in d["frames"] if f["valid"]]
        f_px = (sorted(lens)[len(lens) // 2] / d["sensor_width"] * d["width"]) if lens else 1500.0
        depth = d.get("median_depth") or 10.0 / max(scale, 1e-9)
        _points_modifier(pts, 2.5 * depth / f_px)
        pts.hide_render = True
    scene.frame_set(d["frame_start"])


def _engine_world(d) -> Matrix:
    return Matrix(d.get("world") or Matrix.Identity(4))


def _save_adjust(folder, world, d, **fields):
    """Save the world empty's whole change from the engine's world (size, ground)."""
    adjust = world.matrix_world @ _engine_world(d).inverted()
    solve_io.write_real_scale(folder, adjust=[list(r) for r in adjust], **fields)


def _fit_camera_to_scale(cam_data, world, d):
    """Camera drawing size and clipping for the scene's size (a drone scene in metres reaches
    hundreds of metres; Blender's default clip end is 100)."""
    wscale = max(world.matrix_world.to_scale())
    depth = (d.get("median_depth") or 10.0 / max(wscale, 1e-9)) * wscale  # median depth, scene units
    cam_data.display_size = max(0.05, 0.06 * depth)
    cam_data.clip_start = max(1e-3, depth / 1000.0)
    cam_data.clip_end = max(100.0, depth * 50.0)


def _footage(context, cam_data, d):
    clip_path = bpy.path.abspath(context.scene.niko.clip)
    cam_data.show_background_images = True
    bg = cam_data.background_images.new()
    bg.alpha = 1.0
    bg.display_depth = "BACK"
    # variable frame rate, upright phone video, interlaced: the engine's frames, not the video (Blender
    # would read such a video differently from the solve's frames)
    frames_first = os.path.normpath(os.path.join(context.scene.niko.solve_dir, "frames", d["first_frame_file"]))
    if d.get("footage_frames") and os.path.exists(frames_first):
        clip_path = ""
    if clip_path and os.path.exists(clip_path) and os.path.splitext(clip_path)[1].lower() not in IMAGE_EXT:
        clip = bpy.data.movieclips.load(clip_path, check_existing=True)
        clip.frame_start = d["frame_start"]
        bg.source = "MOVIE_CLIP"
        bg.clip = clip
        return
    # normpath: '//server/share' would be read by Blender as a path relative to the .blend
    first = os.path.normpath(os.path.join(context.scene.niko.solve_dir, "frames", d["first_frame_file"]))
    if clip_path and os.path.exists(clip_path):
        first = os.path.normpath(clip_path)
    if os.path.exists(first):
        img = bpy.data.images.load(first, check_existing=True)
        img.source = "SEQUENCE"
        bg.source = "IMAGE"
        bg.image = img
        bg.image_user.frame_start = d["frame_start"]
        bg.image_user.frame_duration = len(d["frames"])
        bg.image_user.frame_offset = 0


# ----------------------------------------------------------------------------------------- workspace

def setup_workspace(context, switch_back_to=None):
    """Switch to the 'Niko track' workspace: camera view with footage (left), free 3D view with the
    Niko sidebar (right), timeline with the error graph (bottom). Created on first use as a copy of
    Layout. Blender applies workspace changes later, in its event loop, so the layout is arranged
    by a timer once the new workspace is really active (doing it at once crashed Blender 5.2)."""
    win = context.window
    ws = bpy.data.workspaces.get(WORKSPACE)
    if ws is None:
        base = bpy.data.workspaces.get("Layout") or win.workspace
        before = {w.name for w in bpy.data.workspaces}
        with context.temp_override(window=win, workspace=base):
            bpy.ops.workspace.duplicate()
        new = [w for w in bpy.data.workspaces if w.name not in before]
        if not new:
            return None
        ws = new[0]
        ws.name = WORKSPACE
    elif win.workspace != ws:
        win.workspace = ws
    wm = context.window_manager
    idx = list(wm.windows).index(win)
    tries = {"n": 0}

    def _arrange():
        tries["n"] += 1
        wins = bpy.context.window_manager.windows
        if idx >= len(wins) or tries["n"] > 50:
            return None
        w = wins[idx]
        if w.workspace.name != WORKSPACE:
            return 0.1  # not switched yet
        if not w.workspace.get("niko_split"):
            split_workspace(w)
            return 0.3  # let Blender lay the new areas out before arranging them
        arrange_workspace(w)
        if switch_back_to is not None and switch_back_to in bpy.data.workspaces:
            w.workspace = bpy.data.workspaces[switch_back_to]
        return None

    bpy.app.timers.register(_arrange, first_interval=0.1)
    return ws


def split_workspace(win):
    """First visit: split the big 3D view in two and give the timeline room for the error graph."""
    screen = win.screen
    views = [a for a in screen.areas if a.type == "VIEW_3D"]
    if views:
        big = max(views, key=lambda a: a.width * a.height)
        with bpy.context.temp_override(window=win, screen=screen, area=big):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.56)
    tl = [a for a in screen.areas if a.type == "DOPESHEET_EDITOR"]
    if tl:
        a = min(tl, key=lambda a: a.y)
        for dy in (0, 1, 2, -1, 3, 4):  # drag the timeline's top edge up (the edge sits in a small gap)
            with bpy.context.temp_override(window=win, screen=screen, area=a):
                try:
                    r = bpy.ops.screen.area_move(x=a.x + a.width // 2, y=a.y + a.height + dy, delta=110)
                except RuntimeError:
                    r = {"CANCELLED"}
            if "FINISHED" in r:
                break
    win.workspace["niko_split"] = 1


def arrange_workspace(win):
    screen = win.screen
    views = sorted([a for a in screen.areas if a.type == "VIEW_3D"], key=lambda a: a.x)
    if not views:
        return
    left, right = views[0], views[-1]
    sl = left.spaces.active
    sl.region_3d.view_perspective = "CAMERA"
    sl.show_region_ui = False
    sl.show_region_toolbar = False
    sl.overlay.show_floor = False
    sl.overlay.show_axis_x = sl.overlay.show_axis_y = False
    sl.shading.light = "FLAT"  # lock view: points and CG read at full colour over the footage
    sr = right.spaces.active
    sr.show_region_ui = True
    sr.clip_end = max(sr.clip_end, 10000.0)
    _frame_scene(sr.region_3d)
    _run_in(win, left, "WINDOW", bpy.ops.view3d.view_center_camera)
    for a in screen.areas:
        if a.type == "DOPESHEET_EDITOR":
            _run_in(win, a, "WINDOW", bpy.ops.action.view_all)
        a.tag_redraw()


def _frame_scene(rv3d):
    """Aim a free 3D view at the bulk of the points (5-95 %: far outliers would zoom it out)."""
    pts = bpy.data.objects.get("Niko points")
    if pts is None or not len(pts.data.vertices):
        return
    co = np.empty(len(pts.data.vertices) * 3, np.float32)
    pts.data.vertices.foreach_get("co", co)
    M = np.array(pts.matrix_world)
    P = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    lo, hi = np.percentile(P, 5, axis=0), np.percentile(P, 95, axis=0)
    rv3d.view_location = ((lo + hi) / 2).tolist()
    rv3d.view_distance = float(np.linalg.norm(hi - lo)) * 0.6
    cam = bpy.context.scene.camera
    if cam is not None:  # look the way the shot starts
        rv3d.view_rotation = cam.matrix_world.to_quaternion()


def _run_in(win, area, region_type, op, **kw):
    for region in area.regions:
        if region.type == region_type:
            with bpy.context.temp_override(window=win, screen=win.screen, area=area, region=region):
                try:
                    op(**kw)
                except RuntimeError:
                    pass
            return


# ------------------------------------------------------------------------------------------ operators

class NIKO_OT_workspace(bpy.types.Operator):
    bl_idname = "niko.workspace"
    bl_label = "Open Niko workspace"
    bl_description = "Switch to the 'Niko track' workspace (created on first use)"

    def execute(self, context):
        setup_workspace(context)
        return {"FINISHED"}


class NIKO_OT_solve(bpy.types.Operator):
    bl_idname = "niko.solve"
    bl_label = "Solve camera"
    bl_description = "Track the clip and solve its camera with the Niko Tracker Engine (runs in WSL2)"

    _job = None
    _timer = None

    @classmethod
    def poll(cls, context):
        n = context.scene.niko
        return bool(n.clip) and not n.running

    def _refuse(self, n, msg):
        n.error = msg
        n.status = msg
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}

    def execute(self, context):
        n = context.scene.niko
        n.error = ""
        clip = bpy.path.abspath(n.clip).rstrip("\\/")
        if not os.path.exists(clip):
            return self._refuse(n, f"Clip not found: {clip}")
        # what the clip field points at: one video, an image of a sequence, or a folder
        if os.path.isdir(clip):
            files = sorted(os.listdir(clip))
            vids = [f for f in files if os.path.splitext(f)[1].lower() in VIDEO_EXT]
            if any(os.path.splitext(f)[1].lower() in IMAGE_EXT for f in files):
                src = clip
            elif len(vids) == 1:
                src = os.path.join(clip, vids[0])
                n.clip = src
            elif vids:
                return self._refuse(n, f"This folder has {len(vids)} videos ({', '.join(vids[:4])}"
                                       f"{', ...' if len(vids) > 4 else ''}): pick one video file")
            else:
                return self._refuse(n, "No video or image sequence in this folder")
        elif os.path.splitext(clip)[1].lower() in IMAGE_EXT:
            src = os.path.dirname(clip)
        elif os.path.splitext(clip)[1].lower() in VIDEO_EXT:
            src = clip
        else:
            return self._refuse(n, f"Not a video or an image: {os.path.basename(clip)}")
        p = prefs(context)
        try:
            home, _ = niko_home_windows(engine.resolve_distro(p.distro))
        except Exception as e:  # noqa: BLE001 - shown to the user as is
            return self._refuse(n, str(e))
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(src.rstrip("\\/")))[0])
        out = p.solves_dir.replace("$NIKO_HOME", home).rstrip("/") + \
            f"/{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
        args = ["solve", engine.to_wsl(src), "-o", out]
        prompts = n.prompts()
        args += ["--prompts", ",".join(prompts) if prompts else "none"]
        if n.lens_mode == "KNOWN":
            args += ["--focal-mm", f"{n.focal_mm:g}", "--sensor-mm", n.sensor]
        n.reset_stages()
        n.log.clear()
        n.running = True
        n.started = time.time()
        n.stages[0].state, n.stages[0].started = "RUN", n.started
        n.status = "Starting the engine..."
        n.solve_dir = engine.to_windows(out, engine.resolve_distro(p.distro))
        self._out = out
        self._distro = engine.resolve_distro(p.distro)
        try:
            self._job = engine.EngineJob(engine.resolve_distro(p.distro), args)
        except OSError as exc:
            n.running = False
            return self._refuse(n, f"Could not start the engine: {exc}. Check WSL and the engine installation.")
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _advance(self, n, line):
        key = engine.stage_of(line)
        if key is None:
            return
        now = time.time()
        order = [s.key for s in n.stages]
        i = order.index(key)
        for s in n.stages[:i]:
            if s.state in ("WAIT", "RUN"):
                s.state = "DONE"
                s.ended = s.ended or now
        st = n.stages[i]
        if not st.started:
            st.started = now
        if key == "cameras":
            n.candidates_done += 1
            st.detail = f"{n.candidates_done} / {engine.N_CANDIDATES}"
            st.state = "DONE" if n.candidates_done >= engine.N_CANDIDATES else "RUN"
        elif key in ("refine", "select"):
            st.state = "RUN"
        else:
            st.state = "DONE"
        if st.state == "DONE":
            st.ended = now
        if "FAILED" in line:
            st.state = "FAIL"
            n.error = line.split("FAILED:", 1)[-1].strip()[:300]
        nxt = next((s for s in n.stages if s.state == "WAIT"), None)
        if nxt is not None and all(s.state != "RUN" for s in n.stages):
            nxt.state = "RUN"
            nxt.started = now

    def modal(self, context, event):
        n = context.scene.niko
        if event.type == "TIMER":
            for line in self._job.poll_lines():
                ln = n.log.add()
                ln.text = line[-400:]
                if len(n.log) > 400:
                    n.log.remove(0)
                self._advance(n, line)
                n.status = line[-120:]
            if not n.running:  # cancelled
                self._job.cancel()
                return self._finish(context, cancelled=True)
            if self._job.returncode is not None:
                for line in self._job.poll_lines():
                    ln = n.log.add()
                    ln.text = line[-400:]
                    self._advance(n, line)
                return self._finish(context)
            _redraw(context)
        return {"PASS_THROUGH"}

    def _finish(self, context, cancelled=False):
        n = context.scene.niko
        context.window_manager.event_timer_remove(self._timer)
        n.running = False
        if cancelled:
            # wsl.exe going away does not always stop the engine inside WSL: stop it by its output folder
            # (first character in brackets so the pattern does not match this pkill's own shell)
            try:
                engine.query(self._distro, f"pkill -f -- '[{self._out[0]}]{self._out[1:]}'", timeout=20)
            except Exception:  # noqa: BLE001 - best effort
                pass
            n.status = "Cancelled"
            return {"CANCELLED"}
        rc = self._job.returncode
        s = solve_io.get(n.solve_dir, refresh=True)
        if s is None or s.report.get("status") == "failed":
            n.status = f"Failed: {n.error}" if n.error else f"The engine stopped (code {rc}); see the log"
            n.error = n.error or "The solve did not finish. Open Show engine log for details."
            for st in n.stages:
                if st.state == "RUN":
                    st.state = "FAIL"
            _redraw(context)
            return {"CANCELLED"}
        for st in n.stages:
            st.state = "DONE"
        n.error = ""
        build_scene(context, s)
        setup_workspace(context)
        avg = s.average_px
        n.status = "Camera ready - check the result warnings" if s.guidance() else "Camera ready - play the shot and check for sliding"
        _redraw(context)
        return {"FINISHED"}


class NIKO_OT_cancel(bpy.types.Operator):
    bl_idname = "niko.cancel"
    bl_label = "Cancel"
    bl_description = "Stop the running solve"

    def execute(self, context):
        context.scene.niko.running = False
        return {"FINISHED"}


class NIKO_OT_load(bpy.types.Operator):
    bl_idname = "niko.load"
    bl_label = "Load a solve"
    bl_description = "Load a finished solve folder (the folder that holds selected/ and solve.json)"

    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        # start in the engine's solves folder (inside WSL) when it can be found
        try:
            p = prefs(context)
            home, home_win = niko_home_windows(engine.resolve_distro(p.distro))
            solves = p.solves_dir.replace("$NIKO_HOME", home)
            win = engine.to_windows(solves, engine.resolve_distro(p.distro))
            if os.path.isdir(win):
                self.directory = win + os.sep
        except Exception:  # noqa: BLE001 - the browser just opens where Blender last was
            pass
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        folder = self.directory.rstrip("\\/")
        if os.path.basename(folder) == "selected":
            folder = os.path.dirname(folder)
        s = solve_io.get(folder, refresh=True)
        if s is None:
            self.report({"ERROR"}, "No finished solve in this folder (blender.json is missing)")
            return {"CANCELLED"}
        context.scene.niko.solve_dir = folder
        context.scene.niko.clip = s.source_clip() or context.scene.niko.clip
        build_scene(context, s)
        setup_workspace(context)
        return {"FINISHED"}


class NIKO_OT_rebuild(bpy.types.Operator):
    bl_idname = "niko.rebuild"
    bl_label = "Create camera"
    bl_description = "Build the solved camera, footage and points in the scene (replaces the previous ones)"

    @classmethod
    def poll(cls, context):
        return solve_io.get(context.scene.niko.solve_dir) is not None

    def execute(self, context):
        build_scene(context, solve_io.get(context.scene.niko.solve_dir))
        return {"FINISHED"}


class NIKO_OT_jump_worst(bpy.types.Operator):
    bl_idname = "niko.jump_worst"
    bl_label = "Go to worst frame"
    bl_description = "Jump to the frame with the largest tracking error"

    def execute(self, context):
        s = solve_io.get(context.scene.niko.solve_dir)
        f, _ = s.worst_frame() if s else (None, None)
        if f is not None:
            context.scene.frame_set(f)
        return {"FINISHED"}


class NIKO_OT_masks_all(bpy.types.Operator):
    bl_idname = "niko.masks_all"
    bl_label = "Ignore all / none"
    bl_description = "Turn every kind of moving thing on or off (the 'more' field is cleared with None)"

    value: bpy.props.BoolProperty()

    def execute(self, context):
        n = context.scene.niko
        for key in ("person", "car", "animal", "sky", "water"):
            setattr(n, f"ignore_{key}", self.value)
        if not self.value:
            n.ignore_extra = ""
        return {"FINISHED"}


class NIKO_OT_goto_frame(bpy.types.Operator):
    bl_idname = "niko.goto_frame"
    bl_label = "Go to frame"
    bl_description = "Jump to this frame"

    frame: bpy.props.IntProperty()

    def execute(self, context):
        context.scene.frame_set(self.frame)
        return {"FINISHED"}


def _engine_layout(folder: str) -> bool:
    """A folder the engine made (selected/ inside), not a flat copy of selected/."""
    return os.path.isdir(os.path.join(folder, "selected"))


class _EngineTask(bpy.types.Operator):
    """Runs one `niko ...` command on the current solve, then does something with the result.
    On a flat copy of selected/ (e.g. reports/test_shots/01) the engine has no working files:
    `existing` names the file that is opened instead when it is already there."""

    existing = ""

    _job = None
    _timer = None

    @classmethod
    def poll(cls, context):
        n = context.scene.niko
        return solve_io.get(n.solve_dir) is not None and not n.running

    def args(self, solve_wsl):
        raise NotImplementedError

    def done(self, context):
        pass

    def execute(self, context):
        n = context.scene.niko
        if not _engine_layout(n.solve_dir):
            f = os.path.join(solve_io.selected_dir(n.solve_dir), self.existing)
            if self.existing and os.path.exists(f):
                self.done(context)
                return {"FINISHED"}
            self.report({"ERROR"}, "Load the engine's solve folder (the one with selected/ inside) for this")
            return {"CANCELLED"}
        p = prefs(context)
        n.running = True
        n.status = self.bl_label + "..."
        self._job = engine.EngineJob(engine.resolve_distro(p.distro), self.args(engine.to_wsl(n.solve_dir)))
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        n = context.scene.niko
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        for line in self._job.poll_lines():
            ln = n.log.add()
            ln.text = line[-400:]
            n.status = line[-120:]
        if self._job.returncode is None:
            return {"PASS_THROUGH"}
        context.window_manager.event_timer_remove(self._timer)
        n.running = False
        if self._job.returncode != 0:
            n.status = f"{self.bl_label} failed; see the log"
            return {"CANCELLED"}
        self.done(context)
        _redraw(context)
        return {"FINISHED"}


class NIKO_OT_locktest(_EngineTask):
    bl_idname = "niko.locktest"
    bl_label = "Lock test video"
    bl_description = "Render an MP4 of the footage with the solve's points (stay glued = locked) and open it"
    existing = "locktest.mp4"

    def args(self, solve_wsl):
        return ["locktest", solve_wsl, "--scale", "0.5"]

    def done(self, context):
        mp4 = os.path.join(solve_io.selected_dir(context.scene.niko.solve_dir), "locktest.mp4")
        context.scene.niko.status = "Lock test ready"
        if os.path.exists(mp4):
            os.startfile(mp4)  # noqa: S606 - opens the user's own video player


class NIKO_OT_export_ae(_EngineTask):
    bl_idname = "niko.export_ae"
    bl_label = "Send to After Effects"
    bl_description = ("Write a .jsx for After Effects (File > Scripts > Run Script File): comp, footage, "
                      "animated 3D camera and track nulls")
    existing = "niko_after_effects.jsx"

    def args(self, solve_wsl):
        return ["export-ae", solve_wsl]

    def done(self, context):
        jsx = os.path.join(solve_io.selected_dir(context.scene.niko.solve_dir), "niko_after_effects.jsx")
        context.scene.niko.status = "After Effects script ready: run it in AE with File > Scripts > Run Script File"
        if os.path.exists(jsx):
            os.startfile(os.path.dirname(jsx))  # noqa: S606 - shows the file in Explorer


EXPORT_FORMATS = [("FBX", "FBX", "Cinema 4D, Unreal, Maya, 3ds Max (camera with a key on every frame)"),
                  ("ABC", "Alembic", "Houdini, Nuke, Maya, Cinema 4D"),
                  ("USD", "USD", "Houdini Solaris, Omniverse, Unreal, Maya")]


class NIKO_OT_export_3d(bpy.types.Operator):
    bl_idname = "niko.export_3d"
    bl_label = "Export camera"
    bl_description = ("Write the camera (a key on every frame, the lens, the levelled world and real size you set) "
                      "and the track points for another 3D program, next to the solve. Frame 1 of the footage is "
                      "frame 1 there too (time = frame / fps): start the footage at frame 1")
    bl_options = {"REGISTER"}

    fmt: bpy.props.EnumProperty(name="Format", items=EXPORT_FORMATS, default="FBX")

    def execute(self, context):
        n = context.scene.niko
        s = solve_io.get(n.solve_dir)
        cam = bpy.data.objects.get("Niko camera")
        if s is None or cam is None:
            self.report({"ERROR"}, "Load a solve first")
            return {"CANCELLED"}
        names = ["Niko world", "Niko camera", "Niko points", SUN_NAME, CATCHER_NAME]
        obs = [bpy.data.objects[nm] for nm in names if nm in bpy.data.objects]
        prev_sel, prev_act = list(context.selected_objects), context.view_layer.objects.active
        for o in context.view_layer.objects:
            o.select_set(False)
        for o in obs:
            o.select_set(True)
        context.view_layer.objects.active = cam
        out = os.path.join(solve_io.selected_dir(n.solve_dir),
                           "niko_camera." + {"FBX": "fbx", "ABC": "abc", "USD": "usdc"}[self.fmt])
        sc = context.scene
        try:
            if self.fmt == "FBX":
                # the scene's animation baked into one take: with the default "one take per NLA strip"
                # and no strips, Blender 5.2 wrote no animation at all (round-trip check)
                bpy.ops.export_scene.fbx(filepath=out, use_selection=True, bake_anim=True,
                                         bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False,
                                         bake_anim_force_startend_keying=True, bake_anim_step=1.0,
                                         bake_anim_simplify_factor=0.0, object_types={"CAMERA", "EMPTY", "MESH", "LIGHT"},
                                         add_leaf_bones=False, axis_forward="-Z", axis_up="Y")
            elif self.fmt == "ABC":
                bpy.ops.wm.alembic_export(filepath=out, selected=True, start=sc.frame_start, end=sc.frame_end)
            else:
                bpy.ops.wm.usd_export(filepath=out, selected_objects_only=True, export_animation=True)
        except Exception as e:  # noqa: BLE001 - a failed exporter must not leave the selection changed
            self.report({"ERROR"}, f"{self.fmt} export failed: {e}")
            return {"CANCELLED"}
        finally:
            for o in context.view_layer.objects:
                o.select_set(o in prev_sel)
            context.view_layer.objects.active = prev_act
        n.status = f"Camera exported: {os.path.basename(out)} (next to the solve)"
        self.report({"INFO"}, n.status)
        return {"FINISHED"}


def selected_points(context):
    """(object, [world positions]) of the selected vertices of the active mesh (Edit or Object Mode)."""
    import bmesh

    ob = context.active_object
    if ob is None or ob.type != "MESH":
        return None, []
    if ob.mode == "EDIT":
        bm = bmesh.from_edit_mesh(ob.data)
        co = [v.co.copy() for v in bm.verts if v.select]
    else:
        co = [v.co.copy() for v in ob.data.vertices if v.select]
    return ob, [ob.matrix_world @ c for c in co]


class NIKO_OT_set_size(bpy.types.Operator):
    bl_idname = "niko.set_size"
    bl_label = "Set real size"
    bl_description = ("Scale the camera, points and mesh to real size from one known measurement: the "
                      "distance between two selected points, or the camera's height above the ground")
    bl_options = {"REGISTER", "UNDO"}

    mode: bpy.props.EnumProperty(items=[
        ("POINTS", "Two points", "Distance between two selected points ('Niko points' in Edit Mode, or two "
                                 "selected objects such as empties)"),
        ("HEIGHT", "Camera height", "The camera's height above the ground at this frame (drone: the height in "
                                    "its app; handheld: about 1.6 m)")])
    metres: bpy.props.FloatProperty(name="Real value (m)", default=1.0, min=0.001, soft_max=1000.0,
                                    unit="NONE", precision=3)

    def _measure(self, context):
        """(current length in scene units, words for the record) or (None, why not)."""
        if self.mode == "HEIGHT":
            cam = context.scene.camera
            if cam is None:
                return None, "No camera in the scene"
            z = cam.matrix_world.translation.z
            if z <= 1e-6:
                return None, "The camera is not above the ground plane here: use two points instead"
            return z, "camera height"
        ob, pts = selected_points(context)
        if len(pts) != 2:
            sel = [o for o in context.selected_objects if o.type in {"EMPTY", "MESH"}] if ob is None or not pts else []
            if len(sel) == 2:
                pts = [o.matrix_world.translation.copy() for o in sel]
        if len(pts) != 2:
            return None, "Select exactly two points (Edit Mode on 'Niko points') or two objects"
        d = (pts[0] - pts[1]).length
        if d <= 1e-9:
            return None, "The two points are in the same place"
        return d, "two points"

    def invoke(self, context, event):
        length, _ = self._measure(context)
        if length is not None:
            self.metres = round(length, 3)
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        lay = self.layout
        length, why = self._measure(context)
        if length is None:
            lay.label(text=why, icon="ERROR")
            return
        lay.label(text=f"Now: {length:.3f} units ({why})")
        lay.prop(self, "metres")

    def execute(self, context):
        n = context.scene.niko
        world = bpy.data.objects.get("Niko world")
        s = solve_io.get(n.solve_dir)
        length, why = self._measure(context)
        if world is None or s is None:
            self.report({"ERROR"}, "Load a solve first")
            return {"CANCELLED"}
        if length is None:
            self.report({"ERROR"}, why)
            return {"CANCELLED"}
        k = self.metres / length
        world.matrix_world = Matrix.Scale(k, 4) @ world.matrix_world
        cam = bpy.data.objects.get("Niko camera")
        if cam is not None:
            _fit_camera_to_scale(cam.data, world, s.blender)
        old = float(solve_io.read_real_scale(s.folder).get("factor", 1.0))
        how = f"{why} = {self.metres:g} m"
        try:
            _save_adjust(s.folder, world, s.blender, factor=old * k, how=how)
        except OSError as e:
            self.report({"WARNING"}, f"Scaled, but could not save it with the solve: {e}")
        n.status = f"Real size set: {how} (x{k:.4g}). 1 Blender unit = 1 m"
        self.report({"INFO"}, n.status)
        _redraw(context)
        return {"FINISHED"}


def plane_fit(pts):
    """(centre, unit normal, RMS distance of the points from the plane) of 3+ points, or None when
    they are in a line."""
    P = np.array([tuple(p) for p in pts], float)
    c = P.mean(0)
    w, V = np.linalg.eigh((P - c).T @ (P - c) / len(P))
    if w[1] <= 1e-12 * max(w[2], 1e-30):
        return None
    return Vector(c), Vector(V[:, 0]).normalized(), float(np.sqrt(max(w[0], 0.0)))


class NIKO_OT_set_ground(bpy.types.Operator):
    bl_idname = "niko.set_ground"
    bl_label = "Set ground from points"
    bl_description = ("Make the floor you picked the ground: 3 or more selected points on it ('Niko points' in "
                      "Edit Mode, the scene mesh's vertices, or empties) become Z = 0, level, with the origin in "
                      "their middle. Camera height and simulations then use this floor")
    bl_options = {"REGISTER", "UNDO"}

    def _points(self, context):
        ob, pts = selected_points(context)
        if len(pts) < 3:
            sel = [o for o in context.selected_objects if o.type == "EMPTY"]
            if len(sel) >= 3:
                pts = [o.matrix_world.translation.copy() for o in sel]
        return pts

    def execute(self, context):
        n = context.scene.niko
        world = bpy.data.objects.get("Niko world")
        s = solve_io.get(n.solve_dir)
        if world is None or s is None:
            self.report({"ERROR"}, "Load a solve first")
            return {"CANCELLED"}
        pts = self._points(context)
        if len(pts) < 3:
            self.report({"ERROR"}, "Select 3 or more points on the floor (Edit Mode on 'Niko points') or 3 empties")
            return {"CANCELLED"}
        fit = plane_fit(pts)
        if fit is None:
            self.report({"ERROR"}, "The points are in a line: pick points spread over the floor")
            return {"CANCELLED"}
        c, normal, rms = fit
        cam = context.scene.camera
        if cam is not None and normal.dot(cam.matrix_world.translation - c) < 0:
            normal = -normal  # up is the side the camera is on
        size = max((Vector(p) - c).length for p in pts)
        # the smallest turn that levels the floor keeps the scene's heading
        G = normal.rotation_difference(Vector((0, 0, 1))).to_matrix().to_4x4() @ Matrix.Translation(-c)
        world.matrix_world = G @ world.matrix_world
        off = 100 * rms / max(size, 1e-9)
        how = f"{len(pts)} points, {off:.1f}% off flat"
        try:
            _save_adjust(s.folder, world, s.blender, ground=how)
        except OSError as e:
            self.report({"WARNING"}, f"Ground set, but could not save it with the solve: {e}")
        n.status = f"Ground set from {len(pts)} points ({off:.1f}% off a flat plane)"
        if off > 5.0:
            n.status += ": they are not on one flat floor, check them"
        self.report({"INFO"}, n.status)
        _redraw(context)
        return {"FINISHED"}


SUN_NAME = "Niko sun"


class NIKO_OT_add_sun(bpy.types.Operator):
    bl_idname = "niko.add_sun"
    bl_label = "Add the real sun"
    bl_description = ("A sun lamp where the sun really was: from the drone's GPS position and the recording time, "
                      "turned with the scene (north from the GPS track or the compass). 3D objects then cast "
                      "shadows the same way as the footage")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = context.scene.niko
        world = bpy.data.objects.get("Niko world")
        s = solve_io.get(n.solve_dir)
        sun = (s.blender.get("sun") if s else None) or {}
        if world is None or not sun:
            self.report({"ERROR"}, "No sun for this solve (it needs a drone clip with GPS and its recording time)")
            return {"CANCELLED"}
        if sun["elevation_deg"] <= 0:
            n.status = f"The sun was below the horizon ({sun['elevation_deg']:.0f} deg): a night shot"
            self.report({"WARNING"}, n.status)
            return {"CANCELLED"}
        ob = bpy.data.objects.get(SUN_NAME)
        if ob is None:
            light = bpy.data.lights.new(SUN_NAME, "SUN")
            light.energy = 4.0
            light.angle = math.radians(0.53)  # the sun's disc
            ob = bpy.data.objects.new(SUN_NAME, light)
            coll = bpy.data.collections.get(COLLECTION) or context.scene.collection
            coll.objects.link(ob)
        ob.parent = world
        ob.matrix_parent_inverse = Matrix.Identity(4)
        d = Vector(sun["direction"]).normalized()  # towards the sun, in the solve's world
        ob.rotation_mode = "QUATERNION"
        ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)  # a sun lamp shines along its -Z
        ob.location = (0, 0, 0)
        n.status = (f"Sun: azimuth {sun['azimuth_deg']:.0f} deg, {sun['elevation_deg']:.0f} deg high "
                    f"({sun['utc'][:16].replace('T', ' ')} UTC)")
        self.report({"INFO"}, n.status)
        _redraw(context)
        return {"FINISHED"}


CATCHER_NAME = "Niko shadow catcher"


class NIKO_OT_shadow_catcher(bpy.types.Operator):
    bl_idname = "niko.shadow_catcher"
    bl_label = "Add shadow catcher"
    bl_description = ("A ground plane on the scene's floor (Z = 0) that renders only the shadows and reflections "
                      "of your 3D objects (Cycles), with a transparent background, ready to lay over the footage")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = context.scene.niko
        world = bpy.data.objects.get("Niko world")
        s = solve_io.get(n.solve_dir)
        if world is None or s is None:
            self.report({"ERROR"}, "Load a solve first")
            return {"CANCELLED"}
        wscale = max(world.matrix_world.to_scale())
        depth = (s.blender.get("median_depth") or 10.0 / max(wscale, 1e-9)) * wscale  # scene units
        ob = bpy.data.objects.get(CATCHER_NAME)
        if ob is None:
            me = bpy.data.meshes.new(CATCHER_NAME)
            h = 1.0
            me.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
            me.update()
            ob = bpy.data.objects.new(CATCHER_NAME, me)
            coll = bpy.data.collections.get(COLLECTION) or context.scene.collection
            coll.objects.link(ob)
        size = 4.0 * depth
        cam = context.scene.camera
        c = cam.matrix_world.translation if cam is not None else Vector((0, 0, 0))
        ob.matrix_world = Matrix.Translation((c.x, c.y, 0.0)) @ Matrix.Scale(size, 4)
        ob.is_shadow_catcher = True
        context.scene.render.engine = "CYCLES"
        context.scene.render.film_transparent = True
        n.status = f"Shadow catcher on the floor ({2 * size:.3g} units wide). Cycles, transparent background"
        self.report({"INFO"}, n.status)
        _redraw(context)
        return {"FINISHED"}


class NIKO_OT_reset_adjust(bpy.types.Operator):
    bl_idname = "niko.reset_adjust"
    bl_label = "Reset size and ground"
    bl_description = ("Go back to the engine's own size and ground (your setting is kept as "
                      "real_scale.json.old next to the solve)")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        n = context.scene.niko
        world = bpy.data.objects.get("Niko world")
        s = solve_io.get(n.solve_dir)
        if world is None or s is None:
            self.report({"ERROR"}, "Load a solve first")
            return {"CANCELLED"}
        try:
            solve_io.clear_real_scale(s.folder)
        except OSError as e:
            self.report({"ERROR"}, f"Could not reset: {e}")
            return {"CANCELLED"}
        world.matrix_world = _engine_world(s.blender)
        cam = bpy.data.objects.get("Niko camera")
        if cam is not None:
            _fit_camera_to_scale(cam.data, world, s.blender)
        n.status = "Size and ground: back to the engine's"
        _redraw(context)
        return {"FINISHED"}


class NIKO_OT_empties(bpy.types.Operator):
    bl_idname = "niko.empties"
    bl_label = "Empty at selected points"
    bl_description = ("Put an empty on the selected points (Edit Mode on 'Niko points': click points, Shift-click "
                      "for more). Anything parented to it stays locked to the footage")
    bl_options = {"REGISTER", "UNDO"}

    mode: bpy.props.EnumProperty(name="Where", items=[
        ("EACH", "One per point", "An empty on every selected point"),
        ("CENTER", "One in the middle", "One empty at the centre of the selected points "
                                        "(3 points on the floor: an anchor on the floor)")], default="EACH")
    size: bpy.props.FloatProperty(name="Size", default=0.5, min=0.001, soft_max=10.0)

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        return ob is not None and ob.type == "MESH"

    def execute(self, context):
        ob, pts = selected_points(context)
        if not pts:
            self.report({"WARNING"}, "Select points first: 'Niko points' > Tab > click points (Shift-click for more)")
            return {"CANCELLED"}
        if len(pts) > 200 and self.mode == "EACH":
            self.report({"WARNING"}, f"{len(pts)} points selected: pick fewer (up to 200) or use One in the middle")
            return {"CANCELLED"}
        coll = bpy.data.collections.get(COLLECTION) or context.scene.collection
        world = bpy.data.objects.get("Niko world")
        if self.mode == "CENTER":
            from mathutils import Vector
            pts = [sum(pts, Vector()) / len(pts)]
        made = []
        n0 = sum(1 for o in bpy.data.objects if o.name.startswith("Niko point "))
        for i, p in enumerate(pts):
            e = bpy.data.objects.new(f"Niko point {n0 + i + 1:02d}", None)
            e.empty_display_type = "PLAIN_AXES"
            e.empty_display_size = self.size
            coll.objects.link(e)
            e.location = p
            if world is not None:  # keep them with the levelled scene
                e.parent = world
                e.matrix_parent_inverse = world.matrix_world.inverted()
            made.append(e)
        was_edit = ob.mode == "EDIT"
        if was_edit:
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        for e in made:
            e.select_set(True)
        context.view_layer.objects.active = made[-1]
        self.report({"INFO"}, f"{len(made)} empt{'y' if len(made) == 1 else 'ies'} added")
        return {"FINISHED"}


class NIKO_OT_edit_points(bpy.types.Operator):
    bl_idname = "niko.edit_points"
    bl_label = "Pick points"
    bl_description = "Select 'Niko points' and go to Edit Mode (vertex select) to click points in the footage"

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Niko points") is not None

    def execute(self, context):
        pts = bpy.data.objects["Niko points"]
        if context.object is not None and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        pts.hide_set(False)
        pts.select_set(True)
        context.view_layer.objects.active = pts
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="VERT")
        bpy.ops.mesh.select_all(action="DESELECT")
        return {"FINISHED"}


def _version_of(init_py: str):
    """bl_info version tuple from an add-on __init__.py (read as text, not imported)."""
    import ast

    try:
        tree = ast.parse(open(init_py, encoding="utf-8").read())
    except OSError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "bl_info" for t in node.targets):
            info = ast.literal_eval(node.value)
            return tuple(info.get("version", (0, 0, 0)))
    return None


def _install_zip(zf, here: str) -> None:
    """Validate the complete archive and keep the previous installed version."""
    from .updates import install_files, zip_files
    install_files(zip_files(zf), here)


def _download(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "niko-tracker-addon"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _ver(s: str) -> tuple:
    return tuple(int(x) for x in str(s).strip().split("."))


def _update_engine_code(distro: str, info: dict, base: str) -> str:
    """The release's engine code into an installed engine image (installer/engine) when that image
    can run it; returns a note for the status line. A development engine, whose code is the project
    folder itself, has no ENGINE_VERSION and is left alone."""
    import tempfile
    import uuid

    code = info.get("engine_code")
    have = engine.query(distro, 'cat "$NIKO_REPO/ENGINE_VERSION" 2>/dev/null')
    if not code or not have:
        return ""
    image = engine.query(distro, 'cat "$NIKO_HOME/IMAGE_VERSION" 2>/dev/null') or have
    need = info.get("engine_image")
    if need and _ver(need) > _ver(image):
        return f"; the engine itself needs the new setup ({need}) from GitHub"
    if _ver(have) >= _ver(info["version"]):
        return ""
    tmp = os.path.join(tempfile.gettempdir(), code)
    with open(tmp, "wb") as fh:
        fh.write(_download(f"{base}/{code}"))
    ver = str(info["version"])
    stamp = uuid.uuid4().hex
    out = engine.query(distro, (
        f'set -e; E="$NIKO_REPO"; N="$E.new-{stamp}"; B="$E.previous-{stamp}"; '
        '[ -f "$NIKO_HOME/IMAGE_VERSION" ] || cp "$E/ENGINE_VERSION" "$NIKO_HOME/IMAGE_VERSION"; '
        f'mkdir -p "$N"; tar -xzf {engine._q(engine.to_wsl(tmp))} -C "$N"; '
        'test -f "$N/docs/schemas/cameras.schema.json"; '
        'test -f "$N/src/niko/__init__.py"; test -f "$N/src/niko/cli.py"; '
        'test -f "$N/src/niko/pipeline/__init__.py"; '
        'test -f "$N/src/niko/pipeline/solve.py"; '
        'PYTHONPATH="$N/src" "$NIKO_HOME/envs/niko/bin/python" -B -c '
        + engine._q('import niko.cli, niko.pipeline.solve') + '; '
        f'printf "%s\\n" {engine._q(ver)} > "$N/ENGINE_VERSION"; '
        'mv "$E" "$B"; if ! mv "$N" "$E"; then mv "$B" "$E"; exit 1; fi; echo NIKO_OK'), timeout=300)
    os.remove(tmp)
    if not out.endswith("NIKO_OK"):
        raise RuntimeError("Engine update failed validation or installation; check the previous version before retrying")
    return f"; engine code {have} -> {ver} (previous version kept)"


class NIKO_OT_update(bpy.types.Operator):
    bl_idname = "niko.update"
    bl_label = "Update Niko Tracker"
    bl_description = ("Install the newest add-on: from the Niko Tracker project folder when this computer has "
                      "it, from a release folder (latest.json + add-on zip), or else from the GitHub releases. "
                      "Restart Blender afterwards")

    def execute(self, context):
        import io
        import json
        import shutil
        import zipfile

        if context.scene.niko.running:
            self.report({"ERROR"}, "Wait until the solve has finished, then update")
            return {"CANCELLED"}
        p = prefs(context)
        root = p.repo_dir
        note = ""
        here = os.path.dirname(os.path.abspath(__file__))
        cur = _version_of(os.path.join(here, "__init__.py"))
        src = os.path.join(root, "addon", "niko_tracker")
        latest = os.path.join(root, "latest.json")
        try:
            if root and os.path.isfile(os.path.join(src, "__init__.py")):  # the project folder
                new = _version_of(os.path.join(src, "__init__.py"))
                if os.path.normcase(os.path.abspath(src)) == os.path.normcase(here):
                    self.report({"INFO"}, "Blender runs the add-on straight from the project folder: always current")
                    return {"FINISHED"}
                from pathlib import Path
                from .updates import install_files
                install_files({f.name: f.read_bytes() for f in Path(src).glob("*.py")}, here)
            elif root and os.path.isfile(latest):  # a release folder (scripts/publish_release.py)
                with open(latest, encoding="utf-8") as fh:
                    info = json.load(fh)
                new = tuple(int(x) for x in str(info["version"]).split("."))
                with zipfile.ZipFile(os.path.join(root, info["addon_zip"])) as z:
                    _install_zip(z, here)
            else:  # the GitHub releases
                info = json.loads(_download(p.update_url).decode("utf-8"))
                new = _ver(info["version"])
                base = p.update_url.rsplit("/", 1)[0]
                note = _update_engine_code(engine.resolve_distro(p.distro), info, base)
                if cur is not None and new <= cur:
                    context.scene.niko.status = f"Up to date ({'.'.join(map(str, cur))}){note}"
                    self.report({"INFO"}, context.scene.niko.status)
                    return {"FINISHED"}
                with zipfile.ZipFile(io.BytesIO(_download(f"{base}/{info['addon_zip']}"))) as z:
                    _install_zip(z, here)
        except Exception as e:  # network, missing files, a bad zip: say what, change nothing more
            self.report({"ERROR"}, f"Update failed: {type(e).__name__}: {e}")
            return {"CANCELLED"}
        v = ".".join(map(str, new))
        was = ".".join(map(str, cur or ()))
        context.scene.niko.status = (f"Updated {was} -> {v}{note}: restart Blender to use it" if new != cur
                                     else f"Up to date ({v}){note}; files refreshed, restart Blender to reload")
        self.report({"INFO"}, context.scene.niko.status)
        return {"FINISHED"}


MESH_NAME = "Niko scene mesh"


def import_scene_mesh(context, path: str):
    """The engine's mesh.ply (solve world) under 'Niko world', coloured by its vertex colours."""
    old = bpy.data.objects.get(MESH_NAME)
    if old is not None:
        me = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    before = set(bpy.data.objects)
    bpy.ops.wm.ply_import(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    if not new:
        raise RuntimeError("the mesh could not be read")
    ob = new[0]
    ob.name = MESH_NAME
    coll = bpy.data.collections.get(COLLECTION) or context.scene.collection
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    coll.objects.link(ob)
    world = bpy.data.objects.get("Niko world")
    if world is not None:
        ob.parent = world
        ob.matrix_parent_inverse = Matrix.Identity(4)
        ob.matrix_basis = Matrix.Identity(4)
    mat = bpy.data.materials.get(MESH_NAME) or bpy.data.materials.new(MESH_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    col = ob.data.color_attributes.active_color if ob.data.color_attributes else None
    if bsdf is not None and col is not None:
        attr = nt.nodes.get("Niko colour") or nt.nodes.new("ShaderNodeAttribute")
        attr.name = "Niko colour"
        attr.attribute_name = col.name
        attr.location = (-300, 200)
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.9
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    ob.select_set(False)
    pts = bpy.data.objects.get("Niko points")
    if pts is not None:  # the mesh says it better; the points stay one click away in the outliner
        pts.hide_set(True)
    for area in context.screen.areas if context.screen else []:
        if area.type == "VIEW_3D":
            sp = area.spaces.active
            if sp.region_3d is not None and sp.region_3d.view_perspective != "CAMERA":
                for kind in ("VERTEX", "ATTRIBUTE"):  # 5.2 calls it VERTEX, 3.x-4.x ATTRIBUTE
                    try:
                        sp.shading.color_type = kind  # the free view shows the mesh in its own colours
                        break
                    except TypeError:
                        continue
    return ob


PROJ_UV = "Niko projection"


def footage_image(context):
    """The footage as an image datablock (movie or image sequence) for texturing, or None."""
    n = context.scene.niko
    path = bpy.path.abspath(n.clip) if n.clip else ""
    if not path or not os.path.exists(path):
        return None
    img = bpy.data.images.load(path, check_existing=True)
    img.source = "SEQUENCE" if os.path.splitext(path)[1].lower() in IMAGE_EXT else "MOVIE"
    return img


def project_footage(context, on: bool = True):
    """Camera projection: the solved camera throws the footage onto the scene mesh (UV Project
    modifier from 'Niko camera', material with the footage following the frame). on=False goes back
    to the mesh's own vertex colours."""
    ob = bpy.data.objects.get(MESH_NAME)
    cam = bpy.data.objects.get("Niko camera")
    if ob is None or cam is None:
        raise RuntimeError("build the scene mesh first")
    base = bpy.data.materials.get(MESH_NAME)
    if not on:
        mod = ob.modifiers.get(PROJ_UV)
        if mod is not None:
            ob.modifiers.remove(mod)
        ob.data.materials.clear()
        if base is not None:
            ob.data.materials.append(base)
        return None
    img = footage_image(context)
    if img is None:
        raise RuntimeError("set 1 Clip to the video (or first image) first")
    if PROJ_UV not in ob.data.uv_layers:
        ob.data.uv_layers.new(name=PROJ_UV)
    mod = ob.modifiers.get(PROJ_UV) or ob.modifiers.new(PROJ_UV, "UV_PROJECT")
    mod.uv_layer = PROJ_UV
    mod.projector_count = 1
    mod.projectors[0].object = cam
    r = context.scene.render
    mod.aspect_x, mod.aspect_y = r.resolution_x * r.pixel_aspect_x, r.resolution_y * r.pixel_aspect_y
    mat = bpy.data.materials.get("Niko projected footage") or bpy.data.materials.new("Niko projected footage")
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = PROJ_UV
    tex.image = img
    tex.extension = "CLIP"
    iu = tex.image_user
    iu.frame_start = context.scene.frame_start
    iu.frame_duration = context.scene.frame_end - context.scene.frame_start + 1
    iu.use_auto_refresh = True
    uv.location, tex.location, emit.location, out.location = (-700, 0), (-450, 0), (-150, 0), (100, 0)
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    for area in context.screen.areas if context.screen else []:
        if area.type == "VIEW_3D":
            sp = area.spaces.active
            if sp.region_3d is not None and sp.region_3d.view_perspective != "CAMERA":
                sp.shading.type = "MATERIAL"  # the free view shows the projected footage
    return mat


class NIKO_OT_project(bpy.types.Operator):
    bl_idname = "niko.project"
    bl_label = "Project footage on mesh"
    bl_description = ("Camera projection: the solved camera throws the footage onto the scene mesh, so the "
                      "3D scene wears the real pixels (set extension, small camera moves)")
    bl_options = {"REGISTER", "UNDO"}

    on: bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(MESH_NAME) is not None

    def execute(self, context):
        try:
            project_footage(context, self.on)
        except RuntimeError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        context.scene.niko.status = "Footage projected onto the mesh" if self.on else "Mesh in its own colours"
        return {"FINISHED"}


class NIKO_OT_mesh(_EngineTask):
    bl_idname = "niko.mesh"
    bl_label = "Build scene mesh"
    bl_description = ("An editable 3D mesh of the static scene (multi-view stereo on the solved cameras, "
                      "moving things left out). A few minutes; needs a moving camera, not a tripod")
    existing = "mesh.ply"

    def args(self, solve_wsl):
        return ["mesh", solve_wsl, "--quality", bpy.context.scene.niko.mesh_quality.lower()]

    def done(self, context):
        path = os.path.join(solve_io.selected_dir(context.scene.niko.solve_dir), "mesh.ply")
        if not os.path.exists(path):
            context.scene.niko.status = "No mesh was written; see the log"
            return
        ob = import_scene_mesh(context, path)
        context.scene.niko.status = (f"Scene mesh: {len(ob.data.vertices):,} vertices under Niko world"
                                     + ("; a simulation copy is ready (Add simulation collider)"
                                        if os.path.exists(os.path.join(os.path.dirname(path), "mesh_sim.ply"))
                                        else ""))


SIM_NAME = "Niko scene collider"


class NIKO_OT_sim_mesh(bpy.types.Operator):
    bl_idname = "niko.sim_mesh"
    bl_label = "Add simulation collider"
    bl_description = ("The simplified, hole-filled copy of the scene mesh, with a Collision modifier, for physics "
                      "(cloth, particles, rigid bodies, fluids). Hidden in renders")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        s = solve_io.get(context.scene.niko.solve_dir)
        return s is not None and os.path.exists(os.path.join(solve_io.selected_dir(s.folder), "mesh_sim.ply"))

    def execute(self, context):
        path = os.path.join(solve_io.selected_dir(context.scene.niko.solve_dir), "mesh_sim.ply")
        old = bpy.data.objects.get(SIM_NAME)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        before = set(bpy.data.objects)
        bpy.ops.wm.ply_import(filepath=path)
        new = [o for o in bpy.data.objects if o not in before]
        if not new:
            self.report({"ERROR"}, "The simulation mesh could not be read")
            return {"CANCELLED"}
        ob = new[0]
        ob.name = SIM_NAME
        world = bpy.data.objects.get("Niko world")
        if world is not None:
            ob.parent = world
            ob.matrix_parent_inverse = Matrix.Identity(4)
            ob.matrix_basis = Matrix.Identity(4)
        ob.modifiers.new("Collision", "COLLISION")
        ob.display_type = "WIRE"
        ob.hide_render = True
        context.scene.niko.status = f"Simulation collider: {len(ob.data.polygons):,} faces, Collision modifier on"
        self.report({"INFO"}, context.scene.niko.status)
        return {"FINISHED"}


TEX_NAME = "Niko textured mesh"


class NIKO_OT_textured_mesh(bpy.types.Operator):
    bl_idname = "niko.textured_mesh"
    bl_label = "Add textured mesh"
    bl_description = ("The scene mesh with the footage's own pictures as a texture: far fewer triangles than the "
                      "coloured mesh, the same detail, light to render and to edit")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        s = solve_io.get(context.scene.niko.solve_dir)
        return s is not None and os.path.exists(os.path.join(solve_io.selected_dir(s.folder), "mesh_textured.obj"))

    def execute(self, context):
        path = os.path.join(solve_io.selected_dir(context.scene.niko.solve_dir), "mesh_textured.obj")
        old = bpy.data.objects.get(TEX_NAME)
        if old is not None:
            bpy.data.objects.remove(old, do_unlink=True)
        before = set(bpy.data.objects)
        # written in the solve's own axes (not OBJ's Y-up): no conversion
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Y", up_axis="Z")
        new = [o for o in bpy.data.objects if o not in before]
        if not new:
            self.report({"ERROR"}, "The textured mesh could not be read")
            return {"CANCELLED"}
        ob = new[0]
        ob.name = TEX_NAME
        world = bpy.data.objects.get("Niko world")
        if world is not None:
            ob.parent = world
            ob.matrix_parent_inverse = Matrix.Identity(4)
            ob.matrix_basis = Matrix.Identity(4)
        context.scene.niko.status = f"Textured mesh: {len(ob.data.polygons):,} faces with the footage as texture"
        self.report({"INFO"}, context.scene.niko.status)
        return {"FINISHED"}


class NIKO_OT_open_folder(bpy.types.Operator):
    bl_idname = "niko.open_folder"
    bl_label = "Open solve folder"
    bl_description = "Show the solve's files in Explorer"

    @classmethod
    def poll(cls, context):
        return bool(context.scene.niko.solve_dir)

    def execute(self, context):
        os.startfile(context.scene.niko.solve_dir)  # noqa: S606
        return {"FINISHED"}


def _menu_entry(self, context):
    self.layout.separator()
    self.layout.operator("niko.workspace", icon="CAMERA_DATA", text="Niko track")


@bpy.app.handlers.persistent
def _on_load(_dummy=None):
    bpy.app.timers.register(_ensure_tab, first_interval=0.5)


def _ensure_tab():
    """Add the Niko track tab to the open file (preference), leaving the user on their own tab."""
    try:
        p = bpy.context.preferences.addons[__package__].preferences
    except KeyError:
        return None
    wm = bpy.context.window_manager
    if not p.auto_workspace or WORKSPACE in bpy.data.workspaces or not wm.windows:
        return None
    win = wm.windows[0]
    current = win.workspace.name
    with bpy.context.temp_override(window=win):
        setup_workspace(bpy.context, switch_back_to=current)
    return None


_classes = (NIKO_OT_workspace, NIKO_OT_solve, NIKO_OT_cancel, NIKO_OT_load, NIKO_OT_rebuild,
            NIKO_OT_jump_worst, NIKO_OT_goto_frame, NIKO_OT_masks_all, NIKO_OT_set_size, NIKO_OT_set_ground, NIKO_OT_reset_adjust, NIKO_OT_add_sun, NIKO_OT_shadow_catcher, NIKO_OT_export_3d, NIKO_OT_textured_mesh, NIKO_OT_sim_mesh,
            NIKO_OT_locktest, NIKO_OT_export_ae, NIKO_OT_open_folder, NIKO_OT_empties,
            NIKO_OT_edit_points, NIKO_OT_update, NIKO_OT_mesh, NIKO_OT_project)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_workspace_menu.append(_menu_entry)
    bpy.app.handlers.load_post.append(_on_load)
    _on_load()


def unregister():
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    bpy.types.TOPBAR_MT_workspace_menu.remove(_menu_entry)
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
