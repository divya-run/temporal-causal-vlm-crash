from src.sampling.strategies import (
    uniform_sampling,
    impact_centered_sampling,
    phase_based_sampling,
)


def test_uniform_sampling_basic():
    idxs = uniform_sampling(total_frames=100, n=8)
    assert len(idxs) == 8
    assert idxs[0] == 0
    assert idxs[-1] == 99
    assert idxs == sorted(idxs)


def test_uniform_sampling_budget_exceeds_total():
    idxs = uniform_sampling(total_frames=5, n=8)
    assert idxs == [0, 1, 2, 3, 4]


def test_impact_centered_sampling_covers_range():
    idxs = impact_centered_sampling(total_frames=100, n=8, impact_frame=60)
    assert all(0 <= i < 100 for i in idxs)
    assert len(idxs) <= 8


def test_phase_based_sampling_all_phases_present():
    phase_bounds = {
        "pre_crash": (0, 20),
        "conflict": (20, 40),
        "collision": (40, 50),
        "aftermath": (50, 99),
    }
    idxs = phase_based_sampling(total_frames=100, n=8, phase_bounds=phase_bounds)
    assert len(idxs) <= 8
    assert all(0 <= i < 100 for i in idxs)


def test_phase_based_sampling_missing_phases_falls_back():
    idxs = phase_based_sampling(total_frames=100, n=8, phase_bounds={})
    assert len(idxs) == 8  # falls back to uniform_sampling
