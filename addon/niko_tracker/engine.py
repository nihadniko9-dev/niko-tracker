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


# the installer's engine first, then a hand-built one (the development machine's)
ENGINE_DISTROS = ("NikoEngine", "Ubuntu-24.04")
_RESOLVED = {}


def installed_distros() -> list[str]:
    r = subprocess.run(["wsl.exe", "--list", "--quiet"], capture_output=True, timeout=30,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    raw = r.stdout  # UTF-16 unless WSL_UTF8 is set
    text = raw.decode("utf-16-le", "ignore") if b"\x00" in raw else raw.decode("utf-8", "ignore")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def resolve_distro(preferred: str) -> str:
    """The preferred distribution when it is installed, else the first engine distribution that is."""
    if preferred not in _RESOLVED:
        try:
            names = installed_distros()
        except (OSError, subprocess.SubprocessError):
            names = []
        _RESOLVED[preferred] = (preferred if preferred in names
                                else next((d for d in ENGINE_DISTROS if d in names), preferred))
    return _RESOLVED[preferred]


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
