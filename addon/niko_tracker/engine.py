"""Bridge to the Niko Tracker Engine inside WSL2: path mapping, commands, progress parsing."""

import os
import queue
import subprocess
import threading

STAGES = (  # (key, label, log prefixes that mean the stage finished)
    ("ingest", "Read clip", ("[ingest]",)),
    ("masks", "Masks", ("[masks]",)),
    ("tracks", "Tracks", ("[tracks]",)),
    ("cameras", "Camera candidates", ("[colmap_global]", "[colmap_incremental]", "[megasam]", "[da3]")),
    ("refine", "Refine", ("[refine]", "[sift]")),
    ("select", "Pick best", ("[select]", "[solve]")),
    ("export", "Export", ("[export]", "[ae]")),
)
N_CANDIDATES = 4


def to_wsl(path: str) -> str:
    """Windows path -> path inside WSL (D:\\a\\b -> /mnt/d/a/b; \\\\wsl.localhost\\distro\\x -> /x)."""
    p = os.path.abspath(path)
    low = p.lower()
    for prefix in ("\\\\wsl.localhost\\", "\\\\wsl$\\"):
        if low.startswith(prefix):
            rest = p[len(prefix):].split("\\", 1)
            return "/" + (rest[1].replace("\\", "/") if len(rest) > 1 else "")
    if len(p) > 1 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:].replace(os.sep, '/')}"
    return p.replace("\\", "/")


def to_windows(path: str, distro: str) -> str:
    """Path inside WSL -> Windows path (/mnt/d/a -> D:\\a; /home/x -> \\\\wsl.localhost\\distro\\home\\x)."""
    if path.startswith("/mnt/") and len(path) > 6 and path[6] in "/":
        return f"{path[5].upper()}:\\" + path[7:].replace("/", "\\")
    return f"\\\\wsl.localhost\\{distro}" + path.replace("/", "\\")


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def engine_command(distro: str, niko_args: list[str]) -> list[str]:
    """wsl.exe command running `niko <args>` in the engine's environment."""
    inner = ('cd "$NIKO_REPO" && export UV_PROJECT_ENVIRONMENT="$NIKO_HOME/envs/niko" PYTHONUNBUFFERED=1 '
             '&& exec uv run niko ' + " ".join(_q(a) for a in niko_args))
    return ["wsl.exe", "-d", distro, "--exec", "bash", "-lc", inner]


def query(distro: str, shell: str, timeout: float = 60.0) -> str:
    r = subprocess.run(["wsl.exe", "-d", distro, "--exec", "bash", "-lc", shell], capture_output=True,
                       timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.stdout.decode("utf-8", "replace").strip()


class EngineJob:
    """A running `niko ...` process whose output lines are collected on a thread."""

    def __init__(self, distro: str, niko_args: list[str]):
        env = dict(os.environ, WSL_UTF8="1")
        self.proc = subprocess.Popen(engine_command(distro, niko_args), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.lines: "queue.Queue[str]" = queue.Queue()
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self):
        for raw in iter(self.proc.stdout.readline, b""):
            self.lines.put(raw.decode("utf-8", "replace").rstrip())
        self.proc.stdout.close()

    def poll_lines(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                return out

    @property
    def returncode(self):
        return self.proc.poll()

    def cancel(self):
        if self.proc.poll() is None:
            self.proc.terminate()


def stage_of(line: str) -> str | None:
    """The stage a log line reports as finished (lines carry a '[stage]' tag, after an optional '[shot]')."""
    for key, _, prefixes in STAGES:
        if any(p in line for p in prefixes):
            return key
    return None
