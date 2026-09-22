from src.prompting.templates import build_prompt, build_frame_labels


def test_plain_prompt_has_no_frame_labels():
    p = build_prompt("What happened?", ["A car turned", "A car braked"], mode="plain")
    assert p["frame_labels"] is None
    assert "A." in p["user"] and "B." in p["user"]


def test_structured_temporal_labels_frames_by_phase():
    phase_bounds = {"pre_crash": (0, 10), "collision": (11, 20)}
    p = build_prompt(
        "Why did the crash happen?",
        ["Speeding", "Distraction"],
        mode="structured_temporal",
        frame_indices=[2, 15],
        phase_bounds=phase_bounds,
    )
    assert p["frame_labels"] == ["PRE-CRASH", "COLLISION"]


def test_structured_temporal_without_phase_bounds_falls_back():
    labels = build_frame_labels([1, 2, 3], phase_bounds=None)
    assert labels == ["FRAME", "FRAME", "FRAME"]
