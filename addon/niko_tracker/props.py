"""Add-on preferences and per-scene state."""

import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, StringProperty

from .engine import STAGES

# GitHub releases of github.com/nihadniko9-dev/niko-tracker: "latest/download" always points at the
# newest release; latest.json there names the add-on zip (scripts/publish_release.py writes it)
UPDATE_URL = "https://github.com/nihadniko9-dev/niko-tracker/releases/latest/download/latest.json"


class NikoPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    distro: StringProperty(name="WSL distribution", default="NikoEngine",
                           description="The WSL2 distribution that holds the Niko Tracker Engine")
    solves_dir: StringProperty(name="Solves folder (inside WSL)", default="$NIKO_HOME/solves",
                               description="Where the engine writes its working files (Linux file system: fast)")
    auto_workspace: BoolProperty(name="Add the 'Niko track' tab to every file", default=True,
                                 description="Create the Niko track workspace when a file is opened")
    repo_dir: StringProperty(name="Niko Tracker folder", subtype="DIR_PATH", default="D:\\Pack\\Track Nhad",
                             description="Where updates come from: the project folder, or a release folder "
                                         "(latest.json and the add-on zip, e.g. on a shared drive or USB stick)")
    update_url: StringProperty(name="Update address", default=UPDATE_URL,
                               description="latest.json of the GitHub releases: used when this computer has "
                                           "no project folder or release folder")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "distro")
        col.prop(self, "solves_dir")
        col.prop(self, "auto_workspace")
        col.prop(self, "repo_dir")
        col.prop(self, "update_url")
        col.operator("niko.update", icon="FILE_REFRESH")


class NikoStage(bpy.types.PropertyGroup):
    key: StringProperty()
    label: StringProperty()
    started: bpy.props.FloatProperty(default=0.0)
    ended: bpy.props.FloatProperty(default=0.0)
    state: EnumProperty(items=[("WAIT", "Waiting", ""), ("RUN", "Running", ""), ("DONE", "Done", ""),
                               ("FAIL", "Failed", "")], default="WAIT")
    detail: StringProperty()


class NikoLogLine(bpy.types.PropertyGroup):
    text: StringProperty()


class NikoSceneProps(bpy.types.PropertyGroup):
    clip: StringProperty(name="Clip", subtype="FILE_PATH", description="Video file or first image of a sequence")
    ignore_person: BoolProperty(name="Person", default=True)
    ignore_car: BoolProperty(name="Car", default=True)
    ignore_animal: BoolProperty(name="Animal", default=True)
    ignore_sky: BoolProperty(name="Sky", default=True)
    ignore_water: BoolProperty(name="Water", default=True)
    ignore_extra: StringProperty(name="More", description="Other things to ignore, comma separated (e.g. flag, screen)")
    lens_mode: EnumProperty(name="Lens", items=[
        ("AUTO", "Auto", "Measure the lens from the shot"),
        ("KNOWN", "Known", "Use the focal length you know (camera metadata, lens barrel)")], default="AUTO")
    focal_mm: bpy.props.FloatProperty(name="Focal length", default=35.0, min=1.0, max=2000.0, unit="CAMERA",
                                      description="Focal length of the lens in mm")
    sensor: EnumProperty(name="Sensor", default="36.0", items=[
        ("36.0", "Full frame / 35 mm equivalent", "36 mm wide (also phones and drones: use the 35 mm "
         "equivalent focal)"),
        ("24.89", "Super 35", "24.89 mm wide"),
        ("23.5", "APS-C", "23.5 mm wide"),
        ("17.3", "Micro Four Thirds", "17.3 mm wide"),
        ("13.2", "1 inch", "13.2 mm wide")])
    solve_dir: StringProperty(name="Solve folder", subtype="DIR_PATH",
                              description="Result folder of a solve (as Windows sees it)")
    running: BoolProperty(default=False)
    status: StringProperty(default="")
    error: StringProperty(default="")
    started: bpy.props.FloatProperty(default=0.0)
    stages: CollectionProperty(type=NikoStage)
    log: CollectionProperty(type=NikoLogLine)
    candidates_done: IntProperty(default=0)
    show_log: BoolProperty(name="Show engine log", default=False)
    show_points: BoolProperty(name="Show points", default=True)
    show_hud: BoolProperty(name="Show error on camera view", default=True)
    show_graph: BoolProperty(name="Show error graph on timeline", default=True)

    def prompts(self) -> list[str]:
        out = [n for n in ("person", "car", "animal", "sky", "water") if getattr(self, f"ignore_{n}")]
        out += [w.strip() for w in self.ignore_extra.split(",") if w.strip()]
        return out

    def reset_stages(self):
        self.stages.clear()
        for key, label, _ in STAGES:
            s = self.stages.add()
            s.key, s.label, s.state = key, label, "WAIT"
        self.candidates_done = 0


_classes = (NikoPreferences, NikoStage, NikoLogLine, NikoSceneProps)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.niko = bpy.props.PointerProperty(type=NikoSceneProps)


def unregister():
    del bpy.types.Scene.niko
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
