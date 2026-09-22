"""
Prompt construction. `structured_temporal` is the project's main lightweight
intervention (per revised scope): it labels sampled frames with their crash
phase so the model has explicit temporal/causal scaffolding, without any
weight updates.
"""

from __future__ import annotations

PLAIN_SYSTEM_PROMPT = (
    "You are analyzing a short video clip of a roadside traffic incident. "
    "Answer the multiple-choice question using only the provided frames."
)

STRUCTURED_TEMPORAL_SYSTEM_PROMPT = (
    "You are analyzing a short video clip of a roadside traffic incident. "
    "The frames below are labeled with their phase in the incident timeline: "
    "PRE-CRASH (leading up to the event), CONFLICT (developing hazard), "
    "COLLISION (the impact itself), and AFTERMATH (after impact). "
    "Use the phase labels to reason about cause and temporal order before "
    "answering. Answer the multiple-choice question using only the provided frames."
)


def build_question_prompt(question: str, choices: list[str]) -> str:
    lettered = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{lettered}\n\nAnswer with a single letter."


def build_frame_labels(frame_indices: list[int], phase_bounds: dict | None) -> list[str]:
    """
    Map each sampled frame index to a phase label, for structured_temporal
    mode. Falls back to "FRAME" (unlabeled) if phase_bounds is unavailable —
    i.e. structured_temporal degrades to plain prompting for clips without
    phase annotations, which is worth tracking as its own eval slice.
    """
    if not phase_bounds:
        return ["FRAME" for _ in frame_indices]

    labels = []
    for idx in frame_indices:
        label = "FRAME"
        for phase, (start, end) in phase_bounds.items():
            if start <= idx <= end:
                label = phase.replace("_", "-").upper()
                break
        labels.append(label)
    return labels


def build_prompt(
    question: str,
    choices: list[str],
    mode: str,
    frame_indices: list[int] | None = None,
    phase_bounds: dict | None = None,
) -> dict:
    """
    Returns {"system": ..., "user": ..., "frame_labels": [...]} ready to hand
    to the inference wrapper, which interleaves frame_labels with the actual
    image frames when mode == "structured_temporal".
    """
    if mode == "structured_temporal":
        system = STRUCTURED_TEMPORAL_SYSTEM_PROMPT
        frame_labels = build_frame_labels(frame_indices or [], phase_bounds)
    elif mode == "plain":
        system = PLAIN_SYSTEM_PROMPT
        frame_labels = None
    else:
        raise ValueError(f"Unknown prompting mode: {mode}")

    return {
        "system": system,
        "user": build_question_prompt(question, choices),
        "frame_labels": frame_labels,
    }
