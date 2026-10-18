import numpy as np
import pytest

from autosysid.mock import MINI_ARM_XML, MockArm, MockSettings, record_protocol
from autosysid.model import JointDynamics

JOINT = "shoulder_pan"
TRUE_DELAY_S = 0.026
# multipliers of the MJCF nominal values used as "truth" by the mock tests
TRUTH_SCALE = {"kp": 0.7, "kv": 1.5, "damping": 1.4, "frictionloss": 1.5, "armature": 0.8}


@pytest.fixture(scope="session")
def truth() -> JointDynamics:
    nominal = MockArm(MINI_ARM_XML).nominal[JOINT]
    return JointDynamics(**{name: getattr(nominal, name) * factor for name, factor in TRUTH_SCALE.items()})


@pytest.fixture(scope="session")
def make_logs(truth):
    """Factory: logs of the three-part protocol (slow ramps, moderate chirp, saturating steps) on `JOINT`."""

    def factory(seed: int = 0, parts: str = "rcs", scale: float = 1.0):
        settings = MockSettings(seed=seed)
        arm = MockArm(MINI_ARM_XML, truth={JOINT: truth}, delays_s={JOINT: TRUE_DELAY_S}, settings=settings)
        return arm, record_protocol(arm, JOINT, np.random.default_rng(seed), parts=parts, scale=scale)

    return factory
