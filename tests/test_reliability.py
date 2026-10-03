import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from niko import backend
from niko.paths import REPO
from niko.pipeline.cache import check_reuse, signature
from niko.pipeline.solve import SIFT_SPLIT, _finish, _reuse_refined, solve
from test_lens_check import cam


def addon_module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "addon/niko_tracker" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crashed_backend_cannot_read_previous_success(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text('{"ok": true}')
    monkeypatch.setattr(backend, "env_python", lambda name: Path(__file__))
    monkeypatch.setattr(backend, "GpuMemorySampler", MagicMock())
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=9))
    with pytest.raises(backend.BackendError, match="code 9"):
        backend.run_backend("fake", "camera", tmp_path, tmp_path)
    assert not (tmp_path / "result.json").exists()


def test_nonzero_backend_cannot_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "env_python", lambda name: Path(__file__))
    monkeypatch.setattr(backend, "GpuMemorySampler", MagicMock())
    def run(*args, **kwargs):
        (tmp_path / "result.json").write_text('{"ok": true}')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(backend.subprocess, "run", run)
    with pytest.raises(backend.BackendError, match="code 1"):
        backend.run_backend("fake", "camera", tmp_path, tmp_path)


def test_reuse_rejects_changed_content_options_and_failed_run(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"first")
    sig = signature(src, {"focal_mm": 28})
    (tmp_path / "reuse.json").write_text(json.dumps(sig))
    (tmp_path / "solve.json").write_text('{"ok": true}')
    check_reuse(tmp_path, sig)
    with pytest.raises(ValueError, match="Cannot reuse"):
        check_reuse(tmp_path, signature(src, {"focal_mm": 50}))
    src.write_bytes(b"other")  # same file size; content must still invalidate it
    with pytest.raises(ValueError, match="Cannot reuse"):
        check_reuse(tmp_path, signature(src, {"focal_mm": 28}))
    (tmp_path / "solve.json").write_text('{"ok": false}')
    with pytest.raises(ValueError, match="Cannot reuse"):
        check_reuse(tmp_path, sig)


def test_existing_output_is_preserved(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"clip")
    out = tmp_path / "solve"
    out.mkdir()
    (out / "solve.json").write_text("original")
    with pytest.raises(ValueError, match="already contains"):
        solve(src, out)
    assert (out / "solve.json").read_text() == "original"


def test_refined_reuse_requires_the_same_known_lens(tmp_path):
    c = cam(1500)
    opts = {"intrinsics": "fixed"}
    c.extra.update(refine={"options": opts}, sift_split=SIFT_SPLIT, known_focal_px=1500)
    c.save(tmp_path / "cameras.json")
    assert _reuse_refined(tmp_path, "refine", opts, 1500) is not None
    assert _reuse_refined(tmp_path, "refine", opts, 2000) is None
    assert _reuse_refined(tmp_path, "refine", opts) is None


def test_failed_alternative_does_not_invalidate_export(tmp_path):
    stages = {key: {"ok": True} for key in ("ingest", "select", "export", "frame_errors")}
    stages["candidate:megasam"] = {"ok": False, "error": "out of memory"}
    report = _finish({"selected": "colmap+ba", "stages": stages}, tmp_path, 0)
    assert report["ok"] and report["status"] == "completed_with_warnings"
    assert report["warnings"][0]["stage"] == "candidate:megasam"
    stages["export"]["ok"] = False
    assert not _finish(report, tmp_path, 0)["ok"]


def test_small_error_does_not_hide_lens_or_tracking_warnings():
    mod = addon_module("solve_io")
    s = mod.Solve.__new__(mod.Solve)
    s.blender = {"width": 1920, "frames": [{"valid": True}] * 3}
    s.report = {"solve_error": {"average_px": 0.1, "inlier_fraction": 0.5},
                "lens_check": {"uncertain": True}, "track_gaps": [{"from_frame": 1}]}
    advice = s.guidance()
    assert any("Lens uncertain" in x for x in advice)
    assert any("Tracking broke" in x for x in advice)
    assert any("Not reliable" in x for x in advice)  # under 60 % within 3 px (real iPhone 11 night blur: 35 %)
    assert mod.rating(0.1, 1920, 0.5)[0] == "Not reliable"
    s.report["solve_error"]["inlier_fraction"] = 0.75
    assert any("Many tracked" in x for x in s.guidance())
    assert mod.rating(0.1, 1920, 0.75)[0] == "Excellent"


def test_legacy_relative_clip_uses_ingest_source(tmp_path):
    mod = addon_module("solve_io")
    source = tmp_path / "actual.mp4"
    source.write_bytes(b"clip")
    (tmp_path / "shot.json").write_text(json.dumps({"source": str(source)}))
    s = mod.Solve.__new__(mod.Solve)
    s.folder = str(tmp_path)
    s.report = {"clip": "old/relative/clip.mp4"}
    assert s.source_clip() == str(source.resolve())


def test_addon_update_validates_before_writing_and_keeps_backup(tmp_path):
    mod = addon_module("updates")
    here = tmp_path / "niko_tracker"
    here.mkdir()
    (here / "ui.py").write_text("old = True")
    files = {name: b"new = True" for name in mod.REQUIRED}
    with pytest.raises(SyntaxError):
        mod.install_files({**files, "ui.py": b"bad syntax !"}, str(here))
    assert (here / "ui.py").read_text() == "old = True"
    backup = Path(mod.install_files(files, str(here)))
    assert (backup / "ui.py").read_text() == "old = True"
    assert (here / "ui.py").read_text() == "new = True"


def test_addon_update_restores_on_swap_failure(tmp_path, monkeypatch):
    mod = addon_module("updates")
    here = tmp_path / "niko_tracker"
    here.mkdir()
    (here / "ui.py").write_text("old = True")
    original = mod.os.replace
    def replace(src, dst):
        if Path(dst) == here and Path(src).parent.name.startswith("niko-update-"):
            raise OSError("simulated disk failure")
        return original(src, dst)
    monkeypatch.setattr(mod.os, "replace", replace)
    with pytest.raises(OSError, match="simulated"):
        mod.install_files({name: b"new = True" for name in mod.REQUIRED}, str(here))
    assert (here / "ui.py").read_text() == "old = True"


@pytest.mark.parametrize("broken", [False, True])
def test_engine_update_checks_staged_imports_before_swap(tmp_path, broken):
    """Run the actual WSL update shell against a disposable engine, never the installed one."""
    import ast
    import io
    import os
    import shlex
    import subprocess
    import sys
    import tarfile

    if os.name == "nt":
        pytest.skip("engine update runs in Linux")
    home = tmp_path / "home"
    repo = home / "engine"
    repo.mkdir(parents=True)
    (repo / "ENGINE_VERSION").write_text("1.0.0")
    py = home / "envs/niko/bin/python"
    py.parent.mkdir(parents=True)
    py.symlink_to(sys.executable)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        for name in ("docs/schemas/cameras.schema.json", "src/niko/__init__.py", "src/niko/cli.py",
                     "src/niko/pipeline/__init__.py", "src/niko/pipeline/solve.py"):
            content = b"raise RuntimeError('bad release')" if broken and name.endswith("solve.py") else b""
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    def query(distro, shell, timeout=60):
        return subprocess.run(["bash", "-c", shell], capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, "NIKO_HOME": str(home), "NIKO_REPO": str(repo)}).stdout.strip()
    # Isolate this stdlib-only function from bpy while exercising its complete implementation.
    tree = ast.parse((REPO / "addon/niko_tracker/ops.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_update_engine_code")
    scope = {"os": os, "engine": SimpleNamespace(query=query, _q=shlex.quote, to_wsl=lambda p: p),
             "_ver": lambda s: tuple(map(int, s.split("."))), "_download": lambda url: data.getvalue()}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "ops.py", "exec"), scope)
    info = {"engine_code": f"niko-test-{tmp_path.name}.tar.gz", "version": "1.1.0", "engine_image": "1.0.0"}
    if broken:
        with pytest.raises(RuntimeError, match="failed validation"):
            scope["_update_engine_code"]("test", info, "https://example.invalid")
        assert (repo / "ENGINE_VERSION").read_text() == "1.0.0"
    else:
        scope["_update_engine_code"]("test", info, "https://example.invalid")
        assert (repo / "ENGINE_VERSION").read_text().strip() == "1.1.0"
        backup = next(home.glob("engine.previous-*"))
        assert (backup / "ENGINE_VERSION").read_text() == "1.0.0"
