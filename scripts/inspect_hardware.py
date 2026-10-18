"""Read-only inspection of Feetech STS3215 servos on an SO-101 bus.

Phase-0 "inspect before integrating" check (see docs/data-contract.md).
Sends only PING and READ packets: no torque enable, no goal writes, nothing
that can move the arm. Run it in an env that has `scservo_sdk`
(feetech-servo-sdk), e.g. `conda run -n lerobot python scripts/inspect_hardware.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from scservo_sdk import COMM_SUCCESS, GroupSyncRead, PacketHandler, PortHandler

CONFIG = {
    "baudrate": 1_000_000,  # LeRobot's default for STS3215; confirmed against the Baud_Rate register below
    "protocol_end": 0,  # STS/SMS series use the little-endian (end=0) packet variant
    "servo_ids": [1, 2, 3, 4, 5, 6],  # SO-101 follower: shoulder_pan .. gripper
    "n_timing_reads": 300,  # enough samples for stable p50/p99 of the sync-read round trip
    "sync_block_start": 56,  # Present_Position
    "sync_block_len": 8,  # 56..63: position(2) velocity(2) load(2) voltage(1) temperature(1)
}

# (name, address, n_bytes) - addresses taken from lerobot.motors.feetech.tables.STS_SMS_SERIES_CONTROL_TABLE
REGISTERS = [
    ("Firmware_Major_Version", 0, 1),
    ("Firmware_Minor_Version", 1, 1),
    ("Model_Number", 3, 2),
    ("ID", 5, 1),
    ("Baud_Rate", 6, 1),
    ("Return_Delay_Time", 7, 1),
    ("Min_Position_Limit", 9, 2),
    ("Max_Position_Limit", 11, 2),
    ("Max_Torque_Limit", 16, 2),
    ("P_Coefficient", 21, 1),
    ("D_Coefficient", 22, 1),
    ("I_Coefficient", 23, 1),
    ("Protection_Current", 28, 2),
    ("Homing_Offset", 31, 2),
    ("Operating_Mode", 33, 1),
    ("Overload_Torque", 36, 1),
    ("Torque_Enable", 40, 1),
    ("Acceleration", 41, 1),
    ("Goal_Position", 42, 2),
    ("Goal_Velocity", 46, 2),
    ("Torque_Limit", 48, 2),
    ("Lock", 55, 1),
    ("Present_Position", 56, 2),
    ("Present_Velocity", 58, 2),
    ("Present_Load", 60, 2),
    ("Present_Voltage", 62, 1),
    ("Present_Temperature", 63, 1),
    ("Status", 65, 1),
    ("Moving", 66, 1),
    ("Present_Current", 69, 2),
]


def read_register(packet, port, servo_id: int, address: int, n_bytes: int) -> int | None:
    if n_bytes == 1:
        value, comm, _err = packet.read1ByteTxRx(port, servo_id, address)
    else:
        value, comm, _err = packet.read2ByteTxRx(port, servo_id, address)
    return value if comm == COMM_SUCCESS else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/so101-follower")
    parser.add_argument("--out", type=Path, default=Path("docs/hardware_snapshot.json"))
    args = parser.parse_args()

    port = PortHandler(args.port)
    if not port.openPort():
        raise SystemExit(f"cannot open {args.port}")
    port.setBaudRate(CONFIG["baudrate"])
    packet = PacketHandler(CONFIG["protocol_end"])

    snapshot: dict[int, dict[str, int | None]] = {}
    for servo_id in CONFIG["servo_ids"]:
        model, comm, _ = packet.ping(port, servo_id)
        if comm != COMM_SUCCESS:
            print(f"id {servo_id}: NO RESPONSE ({packet.getTxRxResult(comm)})")
            continue
        regs = {name: read_register(packet, port, servo_id, addr, n) for name, addr, n in REGISTERS}
        snapshot[servo_id] = regs
        print(f"\n=== servo {servo_id} (ping model={model}) ===")
        for name, value in regs.items():
            print(f"  {name:<24}{value}")

    # Sync-read round-trip timing over the contiguous state block.
    sync = GroupSyncRead(port, packet, CONFIG["sync_block_start"], CONFIG["sync_block_len"])
    for servo_id in snapshot:
        sync.addParam(servo_id)
    durations_ms: list[float] = []
    failures = 0
    for _ in range(CONFIG["n_timing_reads"]):
        t0 = time.perf_counter()
        comm = sync.txRxPacket()
        durations_ms.append((time.perf_counter() - t0) * 1e3)
        failures += comm != COMM_SUCCESS
    durations_ms.sort()
    print(f"\nsync-read of {len(snapshot)} servos x {CONFIG['sync_block_len']} B, n={len(durations_ms)}")
    print(
        f"  p50={statistics.median(durations_ms):.2f} ms  p99={durations_ms[int(0.99 * len(durations_ms)) - 1]:.2f} ms"
        f"  max={durations_ms[-1]:.2f} ms  failures={failures}"
    )

    port.closePort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
