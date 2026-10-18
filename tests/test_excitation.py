import numpy as np
import pytest

from autosysid.errors import ConfigError
from autosysid.excitation import chirp, multisine, step_train, triangle

TIMES = np.arange(0, 12, 0.01)


def test_chirp_starts_and_ends_at_center_and_respects_amplitude():
    signal = chirp(TIMES, 0.2, 4.0, 0.35, center_rad=0.1)
    assert signal[0] == pytest.approx(0.1) and signal[-1] == pytest.approx(0.1, abs=1e-6)
    assert np.max(np.abs(signal - 0.1)) <= 0.35 + 1e-12


def test_chirp_frequency_increases():
    signal = chirp(TIMES, 0.5, 5.0, 1.0, taper_s=0.0)
    crossings = np.flatnonzero(np.diff(np.sign(signal)) != 0)
    assert np.mean(np.diff(crossings[: len(crossings) // 3])) > np.mean(np.diff(crossings[-len(crossings) // 3 :]))


@pytest.mark.parametrize("args", [(0.0, 1.0), (2.0, 1.0), (1.0, 1.0)])
def test_chirp_rejects_bad_frequencies(args):
    with pytest.raises(ConfigError):
        chirp(TIMES, *args, 0.1)


def test_taper_longer_than_half_the_record_is_rejected():
    with pytest.raises(ConfigError):
        chirp(TIMES, 0.2, 2.0, 0.1, taper_s=7.0)


def test_multisine_is_bounded_and_reproducible():
    frequencies = np.array([0.3, 0.7, 1.9])
    a = multisine(TIMES, frequencies, 0.2, np.random.default_rng(1))
    b = multisine(TIMES, frequencies, 0.2, np.random.default_rng(1))
    assert np.array_equal(a, b) and np.max(np.abs(a)) <= 0.2 + 1e-12


def test_triangle_is_bounded_and_starts_at_center():
    signal = triangle(TIMES, 0.08, 8.0)
    assert signal[0] == pytest.approx(0.0) and np.max(np.abs(signal)) == pytest.approx(0.08, abs=1e-3)


def test_step_train_never_exceeds_slew_limit_and_reaches_levels():
    goal = step_train(TIMES, np.array([0.3, -0.3, 0.0]), 4.0, 0.5)
    assert goal[0] == 0.0
    assert np.max(np.abs(np.diff(goal))) <= 0.5 * 0.01 + 1e-12
    assert goal[int(3.9 / 0.01)] == pytest.approx(0.3)  # reached and held before the next level
