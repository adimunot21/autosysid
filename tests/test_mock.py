import numpy as np
import pytest

from autosysid.errors import ConfigError
from autosysid.excitation import chirp
from autosysid.mock import MINI_ARM_XML, MockArm, MockSettings, jittered_times


def test_jittered_times_are_monotone_and_start_at_zero():
    times = jittered_times(10.0, 100.0, 0.0005, np.random.default_rng(0))
    assert times[0] == 0.0 and np.all(np.diff(times) > 0) and len(times) == 1001


def test_excessive_jitter_is_rejected():
    with pytest.raises(ConfigError):
        jittered_times(1.0, 100.0, 0.05, np.random.default_rng(0))


def test_mock_follows_command_with_quantization_and_delay():
    delay = 0.1
    arm = MockArm(delays_s={"shoulder_pan": delay}, settings=MockSettings(noise_std_rad=0.0, seed=0))
    times = jittered_times(8.0, 100.0, 0.0, np.random.default_rng(0))
    targets = np.zeros((len(times), 3))
    targets[:, 0] = chirp(times, 0.1, 0.8, 0.2, taper_s=1.0)
    log = arm.record(times, targets)
    tick = 2 * np.pi / 4096
    assert np.allclose(log.measured_rad / tick, np.round(log.measured_rad / tick))  # on the encoder grid
    # the response lags the command by about the delay: cross-correlation peak
    lags = np.arange(-30, 31)
    scores = [np.dot(targets[30:-30, 0], np.roll(log.measured_rad[:, 0], -lag)[30:-30]) for lag in lags]
    assert abs(lags[int(np.argmax(scores))] * 0.01 - delay) < 0.03  # measured[t + lag] matches target[t]


def test_mock_is_reproducible_for_a_seed():
    def run():
        arm = MockArm(MINI_ARM_XML, settings=MockSettings(seed=5))
        times = jittered_times(3.0, 100.0, 0.0005, np.random.default_rng(5))
        targets = np.zeros((len(times), 3))
        targets[:, 1] = chirp(times, 0.2, 1.0, 0.1, taper_s=0.5)
        return arm.record(times, targets).measured_rad

    assert np.array_equal(run(), run())


def test_record_validates_shape():
    arm = MockArm()
    with pytest.raises(ConfigError):
        arm.record(np.arange(20) * 0.01, np.zeros((20, 2)))
