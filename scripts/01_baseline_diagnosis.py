"""
Step 1 (unchanged by the professor's revision): reproduce CrashSight's
failure pattern on our model, zero-shot, plain prompting, uniform sampling.

Run:
    python scripts/01_baseline_diagnosis.py --config configs/base.yaml

This establishes the reference point every later frame-budget x
sampling-strategy x prompting-mode combination is compared against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from src.data.crashsight import CrashSightDataset
from src.eval.metrics import RunResult, accuracy, temporal_order_sensitivity
from src.inference.vlm import QwenVLWrapper
from src.prompting.templates import build_prompt
from src.sampling.strategies import uniform_sampling

# Deferred import — only needed once real video decoding is wired up.
def _extract_frames(video_path: Path, frame_indices: list[int]) -> list[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = frame_bgr[:, :, ::-1]
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    dataset = CrashSightDataset(
        root=cfg["data"]["crashsight_root"],
        split=cfg["data"]["split"],
        tiers=cfg["data"]["tiers"],
    )
    print(f"Loaded {len(dataset)} QA items from CrashSight ({cfg['data']['split']} split).")

    model = QwenVLWrapper(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
        device=cfg["model"]["device"],
    )

    frame_budget = cfg["sampling"]["frame_budget"]
    results: list[RunResult] = []

    for item in dataset:
        # NB: total_frames should come from actual video metadata once
        # data is downloaded — placeholder until then.
        import cv2
        cap = cv2.VideoCapture(str(item.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        frame_idxs = uniform_sampling(total_frames, frame_budget)
        frames = _extract_frames(item.video_path, frame_idxs)

        prompt = build_prompt(item.question, item.choices, mode="plain")
        response = model.answer(frames, prompt["system"], prompt["user"])

        correct_letter = chr(65 + item.answer_index)
        results.append(
            RunResult(
                clip_id=item.clip_id,
                tier=item.tier,
                predicted_letter=response.predicted_letter,
                correct_letter=correct_letter,
                latency_ms=response.latency_ms,
                peak_memory_mb=response.peak_memory_mb,
            )
        )

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "overall_accuracy": accuracy(results),
        "per_tier_accuracy": {
            tier: accuracy(results, tier=tier) for tier in cfg["data"]["tiers"]
        },
    }
    (out_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
