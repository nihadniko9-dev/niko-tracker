"""Local fixes for facebookresearch/sam3 @ 2345a4ad (2026-09-18).   python port.py <sam3 checkout>

Idempotent, prints every change.

1. Sam3BasePredictor.start_session() passes offload_state_to_cpu to model.init_state(), but the
   SAM 3.1 multiplex models' init_state() does not take it -> TypeError on every session.
   The multiplex model has no state offloading; accept the argument and refuse True.
"""

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
f = ROOT / "sam3/model/sam3_multiplex_tracking.py"
text = f.read_text()
sig = re.compile(
    r"(    def init_state\(\n        self,\n        resource_path,\n        offload_video_to_cpu=False,\n)"
    r"(        async_loading_frames=False,\n        use_torchcodec=False,\n        use_cv2=False,\n"
    r"        input_is_mp4=False,\n    \):\n)"
)
guard = ("        if offload_state_to_cpu:  # niko port: accepted for the base predictor, not implemented\n"
         "            raise NotImplementedError(\"offload_state_to_cpu is not supported by the multiplex model\")\n")
new, n = sig.subn(lambda m: m.group(1) + "        offload_state_to_cpu=False,\n" + m.group(2) + guard, text)
if n:
    f.write_text(new)
print(f"[port] {f.relative_to(ROOT)}: {n} change(s) - init_state accepts offload_state_to_cpu (False only)")
