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
        if n.status:
            box = lay.box()
            box.scale_y = 0.8
            box.label(text=n.status)


class NIKO_PT_clip(_Base, bpy.types.Panel):
    bl_label = "1  Clip"
    bl_parent_id = "NIKO_PT_main"

    def draw(self, context):
        n = context.scene.niko
        col = self.layout.column()
        col.prop(n, "clip", text="")
        col.operator("niko.load", text="Load a finished solve", icon="FILE_FOLDER")


class NIKO_PT_ignore(_Base, bpy.types.Panel):
    bl_label = "2  Ignore moving things"
    bl_parent_id = "NIKO_PT_main"

    def draw(self, context):
        n = context.scene.niko
        grid = self.layout.grid_flow(row_major=True, columns=3, even_columns=True, align=True)
        for key, icon in (("person", "USER"), ("car", "AUTO"), ("animal", "MONKEY"), ("sky", "LIGHT_SUN"),
                          ("water", "MOD_OCEAN")):
            grid.prop(n, f"ignore_{key}", toggle=True, icon=icon)
        self.layout.prop(n, "ignore_extra", text="", icon="ADD")


class NIKO_PT_solve(_Base, bpy.types.Panel):
    bl_label = "3  Solve"
    bl_parent_id = "NIKO_PT_main"

    def draw(self, context):
        n = context.scene.niko
        lay = self.layout
        row = lay.row(align=True)
        row.prop(n, "lens_mode", expand=True)
        if n.lens_mode == "KNOWN":
            col = lay.column(align=True)
            col.prop(n, "focal_mm")
            col.prop(n, "sensor", text="")
        if n.running:
            row = lay.row()
            row.scale_y = 1.6
            row.operator("niko.cancel", icon="CANCEL")
        else:
            row = lay.row()
            row.scale_y = 1.8
            row.operator("niko.solve", icon="PLAY")
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
    bl_label = "4  Result"
    bl_parent_id = "NIKO_PT_main"

    @classmethod
    def poll(cls, context):
        return solve_io.get(context.scene.niko.solve_dir) is not None

    def draw(self, context):
        s = solve_io.get(context.scene.niko.solve_dir)
        lay = self.layout
        avg = s.average_px
        label, icon, _ = solve_io.rating(avg, s.blender.get("width"))
        box = lay.box()
        row = box.row()
        row.scale_y = 1.6
        row.alert = avg is not None and avg >= 1.0
        k = solve_io.hd_scale(s.blender.get("width"))
        hd = f"  ({avg / k:.2f} HD)" if avg is not None and k > 1 else ""
        row.label(text=f"{avg:.2f} px{hd}   {label}" if avg is not None else label, icon=icon)
        box.label(text="Average error on unseen tracks")
        lc = s.report.get("lens_check") or {}
        kl = s.report.get("known_lens")
        if lc.get("uncertain"):
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
            col.label(text="Know the lens? Use 3 Solve > Known.")
        elif kl:
            lay.label(text=f"Known lens: {kl['focal_mm']:g} mm ({kl['sensor_mm']:g} mm sensor)", icon="LOCKED")
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
            if learned:  # what the depth models read from the images, for judging a lens by eye
                w, sw = s.blender["width"], s.blender["sensor_width"]
                mm = sorted(v / w * sw for v in learned.values())
                _pair(col, "  depth models say", f"{mm[0]:.0f} mm" if mm[-1] - mm[0] < 0.5 else
                      f"{mm[0]:.0f} - {mm[-1]:.0f} mm")
        _pair(col, "Camera", s.camera_kind)
        _pair(col, "Picked", s.selected)
        f, v = s.worst_frame()
        if f is not None:
            r = col.row()
            r.label(text="Worst frame")
            r.operator("niko.jump_worst", text=f"{f}  ({v:.2f} px)", icon="FRAME_NEXT")
        e = s.frame_error(context.scene.frame_current)
        if e is not None:
            _pair(col, "This frame", f"{e:.2f} px")
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
    bl_label = "5  Use it"
    bl_parent_id = "NIKO_PT_main"

    @classmethod
    def poll(cls, context):
        return solve_io.get(context.scene.niko.solve_dir) is not None

    def draw(self, context):
        lay = self.layout
        box = lay.box()
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
        col = lay.column(align=True)
        col.scale_y = 1.3
        col.operator("niko.rebuild", icon="OUTLINER_OB_CAMERA")
        col.operator("niko.mesh", icon="MESH_ICOSPHERE")
        mesh = bpy.data.objects.get("Niko scene mesh")
        if mesh is not None:
            projected = mesh.modifiers.get("Niko projection") is not None
            row = col.row(align=True)
            row.operator("niko.project", text="Project footage on mesh", icon="IMAGE_DATA",
                         depress=projected).on = True
            row.operator("niko.project", text="", icon="COLOR").on = False
        col.operator("niko.locktest", icon="SEQUENCE")
        col.operator("niko.export_ae", icon="EXPORT")
        col.operator("niko.open_folder", icon="FILE_FOLDER")


_classes = (NIKO_PT_main, NIKO_PT_clip, NIKO_PT_ignore, NIKO_PT_solve, NIKO_PT_result, NIKO_PT_use)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
