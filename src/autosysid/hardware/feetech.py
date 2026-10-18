"""`ServoBus` for Feetech STS3215 servos over a serial port, using `scservo_sdk` (package `feetech-servo-sdk`).

Register map and sizes are the ones inspected in Phase 0 (docs/data-contract.md section 1, from
`lerobot.motors.feetech.tables`). NOT yet exercised on a moving arm: until Phase 6 verifies them, signed `velocity` and
`load` decoding follows LeRobot's encoding table (sign bit 15 and 10) and the safety layer does not depend on either.

The port is opened only after checking that no other process holds it (a second reader silently corrupts both
streams), and
nothing is written until the caller does so explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import numpy as np

from autosysid.errors import HardwareError
from autosysid.hardware.bus import SETTINGS, ServoSnapshot
from autosysid.hardware.calibration import ArmCalibration
from autosysid.hardware.clock import Clock, RealClock
from autosysid.logging_utils import get_logger

log = get_logger("feetech")

ADDR_MODEL_NUMBER, ADDR_OPERATING_MODE, ADDR_TORQUE_ENABLE = 3, 33, 40
ADDR_ACCELERATION, ADDR_GOAL_POSITION, ADDR_TORQUE_LIMIT = 41, 42, 48
STATE_BLOCK_ADDRESS, STATE_BLOCK_LENGTH = 56, 8  # position(2) velocity(2) load(2) voltage(1) temperature(1)
STS3215_MODEL_NUMBER = 777
POSITION_MODE = 0
PROTOCOL_END = 0  # STS/SMS series use the little-endian packet variant
VELOCITY_SIGN_BIT, LOAD_SIGN_BIT = 15, 10  # LeRobot STS_SMS_SERIES_ENCODINGS_TABLE (unverified under motion)
_SETTING_REGISTERS = {"torque_limit": (ADDR_TORQUE_LIMIT, 2), "acceleration": (ADDR_ACCELERATION, 1)}
assert set(_SETTING_REGISTERS) == set(SETTINGS)


def decode_sign_magnitude(raw: int, sign_bit: int) -> int:
    """STS registers store negatives as sign-and-magnitude: bit `sign_bit` is the sign, lower bits the magnitude."""
    magnitude = raw & ((1 << sign_bit) - 1)
    return -magnitude if raw & (1 << sign_bit) else magnitude


def port_users(device: str) -> list[tuple[int, str]]:
    """(pid, command) of other processes that hold `device` open. Linux /proc scan; returns [] elsewhere."""
    proc = Path("/proc")
    if not proc.is_dir():
        return []
    target, users = os.path.realpath(device), []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if any(os.path.realpath(fd) == target for fd in (str(p) for p in (entry / "fd").iterdir())):
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                users.append((int(entry.name), command[:100]))
        except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
            continue
    return users


class FeetechBus:
    def __init__(
        self,
        port: str,
        baudrate: int,
        calibration: ArmCalibration,
        joints: tuple[str, ...] | None = None,
        clock: Clock | None = None,
        sdk: ModuleType | None = None,
    ) -> None:
        self.port_name, self.baudrate, self.calibration = port, baudrate, calibration
        self.joints = tuple(joints or calibration.names)
        self.clock: Clock = clock or RealClock()
        self._ids = [calibration.joints[name].servo_id for name in self.joints]
        self._sdk = sdk if sdk is not None else self._import_sdk()
        self._port = self._packet = self._sync_read = self._sync_write = None

    @staticmethod
    def _import_sdk() -> ModuleType:
        try:
            import scservo_sdk
        except ImportError as error:
            raise HardwareError("feetech-servo-sdk is not installed (pip install feetech-servo-sdk)") from error
        return scservo_sdk

    def open(self) -> FeetechBus:
        users = port_users(self.port_name)
        if users:
            raise HardwareError(
                f"{self.port_name} is in use by {users}. Stop that process first; two users corrupt each other's data."
            )
        sdk = self._sdk
        self._port = sdk.PortHandler(self.port_name)
        try:  # the SDK returns False for some failures but lets pyserial's SerialException escape for others
            opened = self._port.openPort()
        except Exception as error:
            self._port = None
            raise HardwareError(f"cannot open {self.port_name}: {error}") from error
        if not opened:
            self._port = None
            raise HardwareError(f"cannot open {self.port_name} (permissions? unplugged?)")
        if not self._port.setBaudRate(self.baudrate):
            self.close()
            raise HardwareError(f"cannot set baudrate {self.baudrate} on {self.port_name}")
        self._packet = sdk.PacketHandler(PROTOCOL_END)
        try:
            for name, servo_id in zip(self.joints, self._ids, strict=True):
                model, comm, _ = self._packet.ping(self._port, servo_id)
                if comm != sdk.COMM_SUCCESS:
                    raise HardwareError(
                        f"{name} (id {servo_id}) did not answer ping: {self._packet.getTxRxResult(comm)}"
                    )
                if model != STS3215_MODEL_NUMBER:
                    raise HardwareError(
                        f"{name} (id {servo_id}) is model {model}, expected STS3215 ({STS3215_MODEL_NUMBER})"
                    )
                mode = self._read(servo_id, ADDR_OPERATING_MODE, 1)
                if mode != POSITION_MODE:
                    raise HardwareError(f"{name} is in operating mode {mode}, expected position mode (0)")
            self._sync_read = sdk.GroupSyncRead(self._port, self._packet, STATE_BLOCK_ADDRESS, STATE_BLOCK_LENGTH)
            for servo_id in self._ids:
                self._sync_read.addParam(servo_id)
            self._sync_write = sdk.GroupSyncWrite(self._port, self._packet, ADDR_GOAL_POSITION, 2)
        except BaseException:
            self.close()
            raise
        log.info("opened %s at %d baud, servos %s", self.port_name, self.baudrate, self._ids)
        return self

    def __enter__(self) -> FeetechBus:
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
            self._port = None

    # ---- ServoBus --------------------------------------------------------------------------------------------

    def read_state(self) -> ServoSnapshot:
        self._require_open()
        started = self.clock.now()
        comm = self._sync_read.txRxPacket()
        ended = self.clock.now()
        if comm != self._sdk.COMM_SUCCESS:
            raise HardwareError(f"sync read failed: {self._packet.getTxRxResult(comm)}")
        columns = []
        for servo_id in self._ids:
            if not self._sync_read.isAvailable(servo_id, STATE_BLOCK_ADDRESS, STATE_BLOCK_LENGTH):
                raise HardwareError(f"servo {servo_id} missing from the sync-read reply")

            def get(offset: int, length: int, sid: int = servo_id) -> int:
                return self._sync_read.getData(sid, STATE_BLOCK_ADDRESS + offset, length)

            columns.append(
                (
                    get(0, 2),
                    decode_sign_magnitude(get(2, 2), VELOCITY_SIGN_BIT),
                    decode_sign_magnitude(get(4, 2), LOAD_SIGN_BIT),
                    get(6, 1) / 10.0,  # Present_Voltage is in 0.1 V
                    get(7, 1),
                )
            )
        position, velocity, load, voltage, temperature = (np.array(values) for values in zip(*columns, strict=True))
        return ServoSnapshot(
            started, ended, position.astype(int), velocity.astype(int), load.astype(int), voltage, temperature
        )

    def write_goal_ticks(self, ticks: np.ndarray) -> None:
        self._require_open()
        ticks = np.asarray(ticks, dtype=int)
        if ticks.shape != (len(self._ids),) or np.any(ticks < 0) or np.any(ticks > 4095):
            raise HardwareError(f"goal ticks {ticks.tolist()} invalid for {len(self._ids)} servos (need 0..4095)")
        for servo_id, value in zip(self._ids, ticks, strict=True):
            if not self._sync_write.addParam(servo_id, [int(value) & 0xFF, int(value) >> 8]):
                self._sync_write.clearParam()
                raise HardwareError(f"could not queue goal for servo {servo_id}")
        comm = self._sync_write.txPacket()
        self._sync_write.clearParam()
        if comm != self._sdk.COMM_SUCCESS:
            raise HardwareError(f"sync write failed: {self._packet.getTxRxResult(comm)}")

    def set_torque(self, enabled: bool) -> None:
        self._require_open()
        errors = []
        for servo_id in self._ids:  # try every servo even if one fails: a half-released arm is the worst case
            try:
                self._write(servo_id, ADDR_TORQUE_ENABLE, 1, int(enabled))
            except HardwareError as error:
                errors.append(str(error))
        if errors:
            raise HardwareError("; ".join(errors))

    def read_setting(self, name: str) -> np.ndarray:
        address, length = _SETTING_REGISTERS[self._setting(name)]
        return np.array([self._read(servo_id, address, length) for servo_id in self._ids], dtype=int)

    def write_setting(self, name: str, values: np.ndarray) -> None:
        address, length = _SETTING_REGISTERS[self._setting(name)]
        for servo_id, value in zip(self._ids, np.asarray(values, dtype=int), strict=True):
            self._write(servo_id, address, length, int(value))

    # ---- register helpers ------------------------------------------------------------------------------------

    def _setting(self, name: str) -> str:
        if name not in _SETTING_REGISTERS:
            raise HardwareError(f"setting '{name}' not allowed; valid: {SETTINGS}")
        self._require_open()
        return name

    def _require_open(self) -> None:
        if self._port is None:
            raise HardwareError("bus is not open")

    def _read(self, servo_id: int, address: int, length: int) -> int:
        read = self._packet.read1ByteTxRx if length == 1 else self._packet.read2ByteTxRx
        value, comm, error = read(self._port, servo_id, address)
        self._raise_if_failed(servo_id, address, comm, error)
        return value

    def _write(self, servo_id: int, address: int, length: int, value: int) -> None:
        write = self._packet.write1ByteTxRx if length == 1 else self._packet.write2ByteTxRx
        comm, error = write(self._port, servo_id, address, value)
        self._raise_if_failed(servo_id, address, comm, error)

    def _raise_if_failed(self, servo_id: int, address: int, comm: int, error: int) -> None:
        if comm != self._sdk.COMM_SUCCESS:
            raise HardwareError(f"servo {servo_id} register {address}: {self._packet.getTxRxResult(comm)}")
        if error != 0:
            raise HardwareError(f"servo {servo_id} register {address}: {self._packet.getRxPacketError(error)}")


if __name__ == "__main__":
    assert decode_sign_magnitude(0x8005, 15) == -5 and decode_sign_magnitude(5, 15) == 5
    assert decode_sign_magnitude(3128, 11) == -1080, "docs/data-contract.md: JSON -1080 <-> register 3128"
    assert decode_sign_magnitude(0x0405, 10) == -5
    print("feetech decode sanity checks passed (no port opened)")
