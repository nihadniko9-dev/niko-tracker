"""Surface-step variants on one solve's fused stereo points (no stereo re-run), each written next to
the mesh as mesh_<name>.ply, for scripts/dev/mesh_check.py.

usage (niko env): python scripts/dev/mesh_variants.py <solve_dir> name=key:value,key:value ...
e.g. nopieces=min_piece:0 coarse=cells:1500 d10=depth:10
"""

import shutil
import sys
from pathlib import Path

from niko.backend import run_backend
from niko.mesh import QUALITY


def main():
    d = Path(sys.argv[1])
    work, sel = d / "mesh", d / "selected"
    base = {k: QUALITY["good"][k] for k in ("cells", "depth", "smooth")}
    for spec in sys.argv[2:]:
        name, _, kv = spec.partition("=")
        opts = dict(base)
        for pair in filter(None, kv.split(",")):
            k, _, v = pair.partition(":")
            opts[k] = float(v) if "." in v else int(v)
        r = run_backend("da3", "mesh", d, work, opts)
        shutil.copyfile(work / "mesh.ply", sel / f"mesh_{name}.ply")
        st = r["stats"]
        print(f"{name}: {opts} -> {st['triangles']:,} triangles, {st.get('components_removed')} pieces removed",
              flush=True)


if __name__ == "__main__":
    main()
