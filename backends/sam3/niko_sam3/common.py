import os
from pathlib import Path

CKPT = Path(os.environ.get("NIKO_HOME", Path.home() / "niko")) / "checkpoints/sam3.1_multiplex/sam3.1_multiplex.pt"


def build_predictor():
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    # FlashAttention 3 is Hopper-only (sm_90); the RTX 5090 is sm_120.
    return build_sam3_multiplex_video_predictor(checkpoint_path=str(CKPT), use_fa3=False)


def masks_for_prompt(predictor, session_id, text: str, n_frames: int, hw=None):
    """Propagate one text prompt through the video. Returns {frame: [N_obj, H, W] bool}, object ids and
    the number of frames SAM 3 gave no output for (filled with "no object").

    SAM 3.1 raises "No points are provided" from its tracker instead of returning empty masks when
    the prompt has nothing left to track in a chunk (real 4K drone clip, frame chunk without
    people): that is taken as "no object" for the frames it did not reach. hw: (H, W) of the frames."""
    import numpy as np

    predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=text))
    per_frame, ids = {}, set()
    try:
        for response in predictor.handle_stream_request(request=dict(type="propagate_in_video",
                                                                      session_id=session_id)):
            out = response["outputs"]
            m = np.asarray(out["out_binary_masks"]).astype(bool)
            per_frame[int(response["frame_index"])] = m
            ids.update(int(i) for i in np.asarray(out["out_obj_ids"]).tolist())
    except RuntimeError as e:
        if "No points are provided" not in str(e):
            raise
    missing = [t for t in range(n_frames) if t not in per_frame]
    if missing:
        shape = next(iter(per_frame.values())).shape[-2:] if per_frame else hw
        if shape is None:
            raise RuntimeError(f"prompt {text!r}: no output for frames {missing[:5]}...")
        for t in missing:
            per_frame[t] = np.zeros((0, *shape), bool)
    return per_frame, sorted(ids), len(missing)
