import numpy as np
import pytest

from autosysid.errors import HardwareError
from autosysid.hardware import feetech
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.feetech import FeetechBus

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex")
CALIBRATION = ArmCalibration.symmetric(JOINTS)


class FakeSDK:
    """Just enough of scservo_sdk to check addresses, widths, byte order and error handling."""

    COMM_SUCCESS = 0

    def __init__(self):
        self.memory = {sid: {} for sid in (1, 2, 3)}
        self.fail_sync_read = False
        for sid in self.memory:
            self.put(sid, 3, 777, 2)
            self.put(sid, 33, 0, 1)
            self.put(sid, 48, 1000, 2)
            self.put(sid, 41, 0, 1)
            self.put(sid, 56, 2048, 2)
            self.put(sid, 62, 120, 1)
            self.put(sid, 63, 27, 1)
        self.log = []
        sdk = self

        class PortHandler:
            def __init__(self, name):
                self.name, self.closed = name, False

            def openPort(self):
                return True

            def setBaudRate(self, baud):
                return True

            def closePort(self):
                self.closed = True

        class PacketHandler:
            def __init__(self, end):
                assert end == 0

            def ping(self, port, sid):
                return sdk.get(sid, 3, 2), 0, 0

            def read1ByteTxRx(self, port, sid, address):
                return sdk.get(sid, address, 1), 0, 0

            def read2ByteTxRx(self, port, sid, address):
                return sdk.get(sid, address, 2), 0, 0

            def write1ByteTxRx(self, port, sid, address, value):
                sdk.put(sid, address, value, 1)
                sdk.log.append(("w1", sid, address, value))
                return 0, 0

            def write2ByteTxRx(self, port, sid, address, value):
                sdk.put(sid, address, value, 2)
                sdk.log.append(("w2", sid, address, value))
                return 0, 0

            def getTxRxResult(self, comm):
                return f"comm error {comm}"

            def getRxPacketError(self, error):
                return f"servo error {error}"

        class GroupSyncRead:
            def __init__(self, port, ph, address, length):
                assert (address, length) == (56, 8)
                self.ids = []

            def addParam(self, sid):
                self.ids.append(sid)
                return True

            def txRxPacket(self):
                return -3 if sdk.fail_sync_read else 0

            def isAvailable(self, sid, address, length):
                return sid in self.ids

            def getData(self, sid, address, length):
                return sdk.get(sid, address, length)

        class GroupSyncWrite:
            def __init__(self, port, ph, address, length):
                assert (address, length) == (42, 2)
                self.params = {}

            def addParam(self, sid, data):
                self.params[sid] = data
                return True

            def txPacket(self):
                for sid, (low, high) in self.params.items():
                    sdk.put(sid, 42, low | (high << 8), 2)
                    sdk.log.append(("sync", sid, [low, high]))
                return 0

            def clearParam(self):
                self.params = {}

        self.PortHandler, self.PacketHandler = PortHandler, PacketHandler
        self.GroupSyncRead, self.GroupSyncWrite = GroupSyncRead, GroupSyncWrite

    def put(self, sid, address, value, length):
        for i in range(length):
            self.memory[sid][address + i] = (value >> (8 * i)) & 0xFF

    def get(self, sid, address, length):
        return sum(self.memory[sid].get(address + i, 0) << (8 * i) for i in range(length))


@pytest.fixture
def sdk():
    return FakeSDK()


@pytest.fixture
def bus(sdk, monkeypatch):
    monkeypatch.setattr(feetech, "port_users", lambda device: [])
    return FeetechBus("/dev/fake", 1_000_000, CALIBRATION, sdk=sdk).open()


def test_open_refuses_a_port_that_another_process_holds(sdk, monkeypatch):
    monkeypatch.setattr(feetech, "port_users", lambda device: [(4242, "python other_project.py")])
    with pytest.raises(HardwareError, match="4242"):
        FeetechBus("/dev/fake", 1_000_000, CALIBRATION, sdk=sdk).open()


def test_open_rejects_wrong_model_and_wrong_mode(sdk, monkeypatch):
    monkeypatch.setattr(feetech, "port_users", lambda device: [])
    sdk.put(2, 3, 1234, 2)
    with pytest.raises(HardwareError, match="expected STS3215"):
        FeetechBus("/dev/fake", 1_000_000, CALIBRATION, sdk=sdk).open()
    sdk.put(2, 3, 777, 2)
    sdk.put(3, 33, 1, 1)
    with pytest.raises(HardwareError, match="position mode"):
        FeetechBus("/dev/fake", 1_000_000, CALIBRATION, sdk=sdk).open()


def test_opening_writes_nothing(sdk, bus):
    assert sdk.log == [], "open() must be read-only: no torque, no goals, no registers"


def test_read_state_decodes_the_documented_block_layout(sdk, bus):
    sdk.put(1, 56, 2500, 2)
    sdk.put(2, 58, 0x8000 | 300, 2)  # negative velocity, sign bit 15
    sdk.put(3, 60, 0x0400 | 50, 2)  # negative load, sign bit 10
    snapshot = bus.read_state()
    assert snapshot.position_ticks.tolist() == [2500, 2048, 2048]
    assert snapshot.velocity_raw.tolist() == [0, -300, 0] and snapshot.load_raw.tolist() == [0, 0, -50]
    assert snapshot.voltage_v.tolist() == [12.0] * 3 and snapshot.temperature_c.tolist() == [27.0] * 3


def test_goal_write_is_little_endian_two_bytes_per_servo(sdk, bus):
    bus.write_goal_ticks(np.array([0x0123, 2047, 4095]))
    assert ("sync", 1, [0x23, 0x01]) in sdk.log and sdk.get(3, 42, 2) == 4095


@pytest.mark.parametrize("ticks", [[-1, 0, 0], [0, 0, 4096], [0, 0]])
def test_goal_write_validates_range_and_shape(bus, ticks):
    with pytest.raises(HardwareError, match="invalid"):
        bus.write_goal_ticks(np.array(ticks))


def test_torque_and_settings_use_the_documented_registers(sdk, bus):
    bus.set_torque(True)
    assert [e for e in sdk.log if e[0] == "w1" and e[2] == 40] == [("w1", sid, 40, 1) for sid in (1, 2, 3)]
    assert bus.read_setting("torque_limit").tolist() == [1000] * 3
    bus.write_setting("torque_limit", np.array([350, 350, 500]))
    assert sdk.get(3, 48, 2) == 500
    bus.write_setting("acceleration", np.array([10, 10, 10]))
    assert sdk.get(1, 41, 1) == 10
    with pytest.raises(HardwareError, match="not allowed"):
        bus.write_setting("Max_Torque_Limit", np.zeros(3))  # EEPROM register: must never be reachable


def test_sync_read_failure_surfaces_as_hardware_error(sdk, bus):
    sdk.fail_sync_read = True
    with pytest.raises(HardwareError, match="sync read failed"):
        bus.read_state()


def test_closed_bus_refuses_io(bus):
    bus.close()
    with pytest.raises(HardwareError, match="not open"):
        bus.read_state()
