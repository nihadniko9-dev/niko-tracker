# Licenses of everything the engine uses

Personal, non-commercial research use only. "Verified" = read from the upstream page on
the date shown; "to verify" = re-checked when the component is installed.

| Component | Used for | License | Status |
|---|---|---|---|
| SAM 3 code (facebookresearch/sam3) | masks | SAM License (Meta, custom) | verified 2026-09-28 |
| SAM 3.1 weights (`facebook/sam3.1`, `sam3.1_multiplex.pt`, gated) | masks | SAM License ("other" on HF) | verified 2026-09-28 |
| CoTracker3 code + weights (facebookresearch/co-tracker) | point tracks | CC-BY-NC 4.0 (some modules Apache-2.0 / MIT) | verified 2026-09-28 |
| COLMAP 4.x (incl. global mapper, ex-GLOMAP) | classic baseline | BSD-3-Clause | to verify |
| ALIKED / LightGlue ONNX models (via COLMAP) | features, matching | BSD-3 / Apache-2.0 | to verify |
| MegaSaM code (mega-sam/mega-sam) | deep SLAM candidate | Apache-2.0 (code), CC-BY 4.0 (other material) | verified 2026-09-28 |
| MegaSaM priors: Depth Anything v1 ViT-L, UniDepth, RAFT | MegaSaM inputs | to verify (Depth Anything V1 Large and UniDepth are believed CC-BY-NC 4.0; RAFT BSD-3) | to verify |
| MegaSaM `megasam_final.pth` (ships in the mega-sam repo) | MegaSaM tracking | follows the repo (Apache-2.0 / CC-BY 4.0) | to verify |
| xformers v0.0.24 NystromAttention behaviour, re-implemented in `backends/megasam/niko_megasam/nystrom.py` | UniDepth inside MegaSaM | BSD-3-Clause (Meta) | noted in the file header |
| Depth Anything 3 code | DA3 candidate | Apache-2.0 | to verify |
| DA3NESTED-GIANT-LARGE-1.1 weights | DA3 candidate | CC-BY-NC 4.0 | verified 2026-09-28 |
| Blender 5.2 | synthetic data, import, lock test | GPL-3.0-or-later | — |
| PyTorch | runtime | BSD-3-Clause | — |
