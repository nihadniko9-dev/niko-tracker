"""The 'Niko' sidebar tab: numbered sections 1 clip, 2 ignore, 3 solve, 4 result, 5 use it."""

import time

import bpy

from . import solve_io

STATE_ICON = {"WAIT": "RADIOBUT_OFF", "RUN": "SORTTIME", "DONE": "CHECKMARK", "FAIL": "CANCEL"}


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Niko"


class NIKO_PT_main(_Base, bpy.types.Panel):
    bl_label = "Niko Tracker"
    bl_options = {"HIDE_HEADER"}

    def draw(self, context):
        lay = self.layout
        row = lay.row(align=True)
        from . import bl_info
        row.label(text=f"Niko Tracker  {'.'.join(map(str, bl_info['version']))}", icon="CAMERA_DATA")
        if context.workspace.name != "Niko track":
            row.operator("niko.workspace", text="", icon="WORKSPACE")
        row.operator("niko.update", text="", icon="FILE_REFRESH")
        n = context.scene.niko
        help_box = lay.box()
        if n.running:
            help_box.label(text="Tracking your camera...", icon="SORTTIME")
            help_box.label(text="You can keep working while this runs.")
        elif solve_io.get(n.solve_dir) is not None:
            help_box.label(text="Next: check your result", icon="INFO")
            help_box.label(text="Play the shot and look for sliding.")
        elif n.clip:
            help_box.label(text="Next: click Solve camera", icon="PLAY")
            help_box.label(text="Automatic settings are ready.")
        else:
            help_box.label(text="Start here: choose your video below", icon="FILE_FOLDER")
            help_box.label(text="For your first try, use a short shot.")
        if n.status and (not n.running or n.show_log):
            box = lay.box()
            box.scale_y = 0.8
            box.label(text=n.status)


class NIKO_PT_clip(_Base, bpy.types.Panel):
    bl_label = "1  Choose your video"
    bl_parent_id = "NIKO_PT_main"

    def draw(self, context):
        n = context.scene.niko
        col = self.layout.column()
        col.prop(n, "clip", text="")
        col.label(text="One continuous shot, without cuts.")
        col.operator("niko.load", text="Load a finished solve", icon="FILE_FOLDER")


MASK_KINDS = (("person", "USER"), ("car", "AUTO"), ("animal", "MONKEY"), ("sky", "LIGHT_SUN"),
              ("water", "MOD_OCEAN"))


class NIKO_PT_ignore(_Base, bpy.types.Panel):
    """Chosen before solving: mask nothing, some or all moving things (the switch in the header)."""
    bl_label = "Ignore moving things"
    bl_parent_id = "NIKO_PT_main"

    def draw_header(self, context):
        self.layout.enabled = not context.scene.niko.running
        self.layout.prop(context.scene.niko, "use_masks", text="")

    def draw(self, context):
        n = context.scene.niko
        lay = self.layout
        lay.enabled = not n.running
        if not n.use_masks:
            col = lay.column(align=True)
            col.scale_y = 0.8
            col.label(text="Off: every pixel is tracked.", icon="INFO")
            col.label(text="Faster; use it when nothing moves.")
            return
        row = lay.row(align=True)
        row.operator("niko.masks_all", text="All", icon="CHECKBOX_HLT").value = True
        row.operator("niko.masks_all", text="None", icon="CHECKBOX_DEHLT").value = False
        grid = lay.grid_flow(row_major=True, columns=3, even_columns=True, align=True)
        for key, icon in MASK_KINDS:
            grid.prop(n, f"ignore_{key}", toggle=True, icon=icon)
        lay.prop(n, "ignore_extra", text="", icon="ADD")
        if not n.prompts():
            lay.label(text="Nothing chosen: nothing is ignored.", icon="INFO")


class NIKO_PT_solve(_Base, bpy.types.Panel):
    bl_label = "2  Solve camera"
    bl_parent_id = "NIKO_PT_main"

    def draw(self, context):
        n = context.scene.niko
        lay = self.layout
        settings = lay.column()
        settings.enabled = not n.running
        settings.prop(n, "advanced")
        if n.advanced:
            row = settings.row(align=True)
            row.prop(n, "lens_mode", expand=True)
            if n.lens_mode == "KNOWN":
                col = settings.column(align=True)
                col.prop(n, "focal_mm")
                col.prop(n, "sensor", text="")
        else:
            settings.label(text="Lens: automatic" if n.lens_mode == "AUTO" else f"Known lens: {n.focal_mm:g} mm")
            chosen = n.prompts()
            settings.label(text=("Ignoring: " + ", ".join(chosen))[:44] if chosen else "Ignoring: nothing")
        if n.running:
            row = lay.row()
            row.scale_y = 1.6
            row.operator("niko.cancel", icon="CANCEL")
        else:
            row = lay.row()
            row.scale_y = 1.8
            row.operator("niko.solve", text="Solve camera", icon="PLAY")
        if n.error and not n.running:
            box = lay.box()
            box.alert = True
            col = box.column(align=True)
            col.label(text="Could not solve", icon="ERROR")
            for chunk in _wrap(n.error, 44):
                col.label(text=chunk)
        if len(n.stages):
            col = lay.column(align=True)
            now = time.time()
            for s in n.stages:
                r = col.row()
                r.alert = s.state == "FAIL"
                r.label(text=s.label, icon=STATE_ICON[s.state])
                right = s.detail
                if s.started and s.state in ("RUN", "DONE"):
                    t = (now if s.state == "RUN" or not s.ended else s.ended) - s.started
                    right = (right + "  " if right else "") + _clock(t)
                if right:
                    r.label(text=right)
            if n.running and n.started:
                col.label(text=f"Total {_clock(now - n.started)}   (long 4K clips take several minutes a stage)")
        lay.prop(n, "show_log", toggle=False)
        if n.show_log and len(n.log):
            box = lay.box()
            col = box.column(align=True)
            col.scale_y = 0.7
            for ln in list(n.log)[-14:]:
                col.label(text=ln.text)


class NIKO_PT_result(_Base, bpy.types.Panel):
    bl_label = "3  Check your result"
    bl_parent_id = "NIKO_PT_main"

    @classmethod
    def poll(cls, context):
        return solve_io.get(context.scene.niko.solve_dir) is not None

    def draw(self, context):
        s = solve_io.get(context.scene.niko.solve_dir)
        lay = self.layout
        avg = s.average_px
        advice = s.guidance()
        frac = (s.report.get("solve_error") or {}).get("inlier_fraction")
        if frac is not None and frac < solve_io.UNRELIABLE_FRACTION:
            label, icon = "Not reliable", "CANCEL"
        else:
            label, icon = ("Needs review", "ERROR") if advice else ("Ready for visual check", "INFO")
        box = lay.box()
        row = box.row()
        row.scale_y = 1.6
        row.alert = bool(advice)
        k = solve_io.hd_scale(s.blender.get("width"))
        hd = f"  ({avg / k:.2f} HD)" if avg is not None and k > 1 else ""
        row.label(text=label, icon=icon)
        box.label(text=f"Inlier average: {avg:.2f} px{hd}" if avg is not None else "Error not measured")
        box.label(text="Average includes errors below 3 px.")
        fraction = (s.report.get("solve_error") or {}).get("inlier_fraction")
        if fraction is not None:
            box.label(text=f"Observations within 3 px: {100 * fraction:.1f}%")
        for message in advice:
            for chunk in _wrap(message, 42):
                lay.label(text=chunk)
        lay.operator("niko.locktest", text="Make a lock-test video", icon="SEQUENCE")
        lay.label(text="Watch for points sliding on the footage.")
        lc = s.report.get("lens_check") or {}
        kl = s.report.get("known_lens")
        if lc.get("uncertain") and context.scene.niko.advanced:
            warn = lay.box()
            warn.alert = True
            warn.label(text="Lens uncertain", icon="ERROR")
            col = warn.column(align=True)
            col.scale_y = 0.8
            col.label(text="This move can't measure the lens.")
            sp = lc.get("spread") or {}
            if "equally_good_fits" in (lc.get("reasons") or []) and sp.get("focal_px"):
                w, sw = s.blender["width"], s.blender["sensor_width"]
                lo, hi = (v / w * sw for v in sp["focal_px"])
                col.label(text=f"Equally good fits: {lo:.0f} - {hi:.0f} mm")
            col.label(text="Use 2 Solve > Advanced > Known.")
        elif kl:
            # given: mm on a sensor; Sony metadata: mm and its 35 mm equivalent; maker's spec: the
            # equivalent only; camera profile: pixels only
            if kl.get("focal_mm") and kl.get("sensor_mm"):
                text = f"Known lens: {kl['focal_mm']:g} mm ({kl['sensor_mm']:g} mm sensor)"
            elif kl.get("focal_mm") and kl.get("focal_35mm"):
                text = f"Known lens: {kl['focal_mm']:g} mm ({kl['focal_35mm']:.0f} equiv.)"
            elif kl.get("focal_35mm"):
                text = f"Known lens: {kl['focal_35mm']:g} mm equivalent"
            else:
                w, sw = s.blender.get("width"), s.blender.get("sensor_width")
                text = f"Known lens: {kl['focal_px'] / w * sw:.1f} mm" if w and sw and kl.get("focal_px") else "Known lens"
            lay.label(text=text, icon="LOCKED")
        gaps = s.report.get("track_gaps") or []
        if gaps:
            warn = lay.box()
            warn.alert = True
            warn.label(text="Tracking broke", icon="ERROR")
            col = warn.column(align=True)
            col.scale_y = 0.8
            for g in gaps[:4]:
                r = col.row()
                r.label(text=f"Frames {g['from_frame']} - {g['to_frame']}")
                r.operator("niko.goto_frame", text="", icon="FRAME_NEXT").frame = g["from_frame"]
            col.label(text="Fast move or blur: nothing ties")
            col.label(text="the two sides, the turn between")
            col.label(text="them can be off by a few degrees.")
        col = lay.column(align=True)
        done, total = s.frames_solved
        lo, hi = s.lens_mm
        _pair(col, "Frames solved", f"{done} / {total}")
        if lo is not None:
            _pair(col, "Lens", f"{lo:.1f} mm" if abs(hi - lo) < 0.05 else f"{lo:.1f} - {hi:.1f} mm")
            learned = (s.report.get("lens_check") or {}).get("learned_focal_px") or {}
            if learned and context.scene.niko.advanced:
                w, sw = s.blender["width"], s.blender["sensor_width"]
                mm = sorted(v / w * sw for v in learned.values())
                _pair(col, "  depth models say", f"{mm[0]:.0f} mm" if mm[-1] - mm[0] < 0.5 else
                      f"{mm[0]:.0f} - {mm[-1]:.0f} mm")
        _pair(col, "Camera", s.camera_kind)
        made = (s.report.get("camera") or {}).get("name")
        if made:
            _pair(col, "Shot with", made.replace("DJI DJI ", "DJI "))
        known = s.report.get("known_lens") or {}
        if known.get("source") in ("camera profile", "camera metadata", "maker's lens spec"):
            _pair(col, "Lens from", known["source"])
        if context.scene.niko.advanced:
            _pair(col, "Picked", s.selected)
        f, v = s.worst_frame()
        if f is not None:
            r = col.row()
            r.label(text="Worst frame")
            r.operator("niko.jump_worst", text=f"{f}  ({v:.2f} px)", icon="FRAME_NEXT")
        e = s.frame_error(context.scene.frame_current)
        if e is not None:
            _pair(col, "This frame", f"{e:.2f} px")
        failed = [(k, v) for k, v in s.report.get("stages", {}).items() if not v.get("ok")]
        if failed and context.scene.niko.advanced:
            details = lay.box()
            details.label(text="Processing details", icon="INFO")
            for name, value in failed:
                details.label(text=name)
                for chunk in _wrap(value.get("error", "Failed")[:180], 42):
                    details.label(text=chunk)
        lay.separator()
        row = lay.row(align=True)
        row.prop(context.scene.niko, "show_hud", text="", icon="FONT_DATA")
        row.prop(context.scene.niko, "show_graph", text="", icon="GRAPH")
        row.label(text="Error overlays")


def _clock(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 60}:{s % 60:02d}"


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for w in text.split():
        if len(line) + len(w) + 1 > width and line:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return out + ([line] if line else [])


def _pair(col, a, b):
    r = col.row()
    r.label(text=a)
    r.label(text=b)


class NIKO_PT_use(_Base, bpy.types.Panel):
    bl_label = "Use your camera"
    bl_parent_id = "NIKO_PT_main"

    @classmethod
    def poll(cls, context):
        return solve_io.get(context.scene.niko.solve_dir) is not None

    def draw(self, context):
        lay = self.layout
        box = lay.box()
        box.label(text="Place your 3D object", icon="INFO")
        box.label(text="Anchor something to the footage", icon="EMPTY_AXIS")
        ob = context.active_object
        editing = ob is not None and ob.name == "Niko points" and ob.mode == "EDIT"
        row = box.row(align=True)
        row.scale_y = 1.2
        if editing:
            row.operator("niko.empties", text="Empty at each point", icon="EMPTY_AXIS").mode = "EACH"
            row.operator("niko.empties", text="One in the middle", icon="PIVOT_MEDIAN").mode = "CENTER"
            box.label(text="Click points, Shift-click for more. Tab to leave.", icon="INFO")
        else:
            row.operator("niko.edit_points", text="Pick points", icon="RESTRICT_SELECT_OFF")
        s = solve_io.get(context.scene.niko.solve_dir)
        size = lay.box()
        size.label(text="Real size (metres)", icon="DRIVER_DISTANCE")
        for chunk in _wrap(s.size_text(), 42):
            size.label(text=chunk)
        row = size.row(align=True)
        row.operator("niko.set_size", text="From two points", icon="DRIVER_DISTANCE").mode = "POINTS"
        row.operator("niko.set_size", text="Camera height", icon="OUTLINER_OB_CAMERA").mode = "HEIGHT"
        for chunk in _wrap(s.ground_text(), 42):
            size.label(text=chunk)
        row = size.row(align=True)
        row.operator("niko.set_ground", icon="AXIS_TOP")
        row.operator("niko.reset_adjust", text="", icon="LOOP_BACK")
        sun = s.blender.get("sun")
        if sun:
            row = size.row()
            if sun["elevation_deg"] > 0:
                row.operator("niko.add_sun", icon="LIGHT_SUN")
                size.label(text=f"Sun {sun['azimuth_deg']:.0f}\u00b0, {sun['elevation_deg']:.0f}\u00b0 high, "
                                f"{sun['utc'][11:16]} UTC")
            else:
                size.label(text="Night shot: the sun was below the horizon", icon="LIGHT_SUN")
        col = lay.column(align=True)
        col.scale_y = 1.3
        col.operator("niko.rebuild", icon="OUTLINER_OB_CAMERA")
        row = col.row(align=True)
        row.prop(context.scene.niko, "mesh_quality", text="")
        row.operator("niko.mesh", icon="MESH_ICOSPHERE")
        col.label(text="Scene mesh is optional.")
        col.operator("niko.textured_mesh", icon="TEXTURE")
        col.operator("niko.sim_mesh", icon="PHYSICS")
        col.operator("niko.shadow_catcher", icon="SHADING_SOLID")
        mesh = bpy.data.objects.get("Niko scene mesh")
        if mesh is not None:
            projected = mesh.modifiers.get("Niko projection") is not None
            row = col.row(align=True)
            row.operator("niko.project", text="Project footage on mesh", icon="IMAGE_DATA",
                         depress=projected).on = True
            row.operator("niko.project", text="", icon="COLOR").on = False
        col.operator("niko.locktest", icon="SEQUENCE")
        col.operator("niko.export_ae", icon="EXPORT")
        row = col.row(align=True)
        for key, label, _ in (("FBX", "FBX", ""), ("ABC", "Alembic", ""), ("USD", "USD", "")):
            row.operator("niko.export_3d", text=label, icon="EXPORT").fmt = key
        col.operator("niko.open_folder", icon="FILE_FOLDER")


_classes = (NIKO_PT_main, NIKO_PT_clip, NIKO_PT_ignore, NIKO_PT_solve, NIKO_PT_result, NIKO_PT_use)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
