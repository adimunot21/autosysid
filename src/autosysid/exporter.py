"""Write fitted parameters back into the user's MJCF with minimal, verifiable edits.

Why text surgery and not `MjSpec.to_xml()`: `to_xml()` re-emits the *compiled* model - comments, `<include>` structure,
`<default class>` layout and attribute order are lost, so the user could not review or merge the change. Instead we
locate the exact start tags of `<joint name=X>` and `<position name=X>` (expat gives byte offsets) and change only
their attributes. Everything else is byte-identical.

Inheritance (docs/data-contract.md section 2): in the SO-ARM100 model the values live in `<default class="sts3215">`
shared by every joint. We never edit the class; we write explicit attributes on the individual elements, which override
the class. Files that `<include>` a patched file are re-emitted with the include path rewritten, so the originals are
never touched and the calibrated model is self-consistent.

Every write is verified by recompiling the output and reading the values back (`model.read_dynamics`).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from xml.parsers import expat

import mujoco

from autosysid import __version__
from autosysid.config import FitConfig
from autosysid.errors import DataError, ExportError
from autosysid.logging_utils import get_logger
from autosysid.model import DYNAMICS_FIELDS, load_spec, read_dynamics
from autosysid.optimizer import JointFit

log = get_logger("exporter")

CALIBRATION_SCHEMA_VERSION = 1
_JOINT_ATTRIBUTES = ("damping", "frictionloss", "armature")  # <joint> attributes; kp/kv go on the <position> actuator
_NOT_JOINT_PARENTS = {"default", "equality"}  # <joint> with no identity (class defaults) or a constraint reference
_READBACK_RELATIVE_TOLERANCE = 1e-6  # values are written with 9 significant digits


@dataclass(frozen=True)
class _Tag:
    """A start tag located in a file: byte span plus parsed attributes."""

    name: str
    parent: str
    attributes: dict[str, str]
    start: int
    end: int  # exclusive; just past '>'


@dataclass(frozen=True)
class ExportedJoint:
    joint: str
    exported: bool
    reason: str  # why it was skipped, or "" when exported
    written: dict[str, float] = field(default_factory=dict)  # MJCF attributes written (kp, kv, damping, ...)


@dataclass(frozen=True)
class ExportResult:
    xml_paths: tuple[Path, ...]  # main file first, then any re-emitted included files
    calibration_path: Path
    joints: tuple[ExportedJoint, ...]


def _scan_file(path: Path) -> tuple[bytes, list[_Tag]]:
    """Parse `path` and return its bytes and every start tag with byte offsets."""
    data = path.read_bytes()
    tags: list[_Tag] = []
    stack: list[str] = []
    parser = expat.ParserCreate()

    def on_start(name: str, attributes: dict[str, str]) -> None:
        start = parser.CurrentByteIndex
        tags.append(_Tag(name, stack[-1] if stack else "", dict(attributes), start, _tag_end(data, start)))
        stack.append(name)

    parser.StartElementHandler = on_start
    parser.EndElementHandler = lambda name: stack.pop()
    try:
        parser.Parse(data, True)
    except expat.ExpatError as error:
        raise ExportError(f"cannot parse {path}: {error}") from error
    return data, tags


def _tag_end(data: bytes, start: int) -> int:
    """Index just past the '>' closing the tag that begins at `start` ('>' inside quoted values does not count)."""
    quote = None
    for index in range(start, len(data)):
        char = data[index : index + 1]
        if quote:
            quote = None if char == quote else quote
        elif char in (b'"', b"'"):
            quote = char
        elif char == b">":
            return index + 1
    raise ExportError("unterminated tag")


def _format_number(value: float) -> str:
    return f"{value:.9g}"


def _set_attribute(tag_text: str, attribute: str, value: float) -> str:
    """Replace `attribute` in a start tag, or append it before the closing '>' / '/>'. Other text is preserved."""
    new_value = _format_number(value)
    pattern = re.compile(rf"(?<=\s){attribute}(\s*=\s*)(\"[^\"]*\"|'[^']*')")
    match = pattern.search(tag_text)
    if match:
        quote = match.group(2)[0]
        old_numbers = match.group(2)[1:-1].split()
        # `damping` may be a polynomial "c0 c1 c2" in MuJoCo 3.16: only c0 is linear damping, keep the rest
        replaced = " ".join([new_value, *old_numbers[1:]]) if attribute == "damping" else new_value
        return (
            tag_text[: match.start()] + f"{attribute}{match.group(1)}{quote}{replaced}{quote}" + tag_text[match.end() :]
        )
    closer = "/>" if tag_text.endswith("/>") else ">"
    body = tag_text[: -len(closer)]
    stripped = body.rstrip()
    return f'{stripped} {attribute}="{new_value}"{body[len(stripped) :]}{closer}'


def _resolve_includes(main_path: Path) -> dict[Path, tuple[bytes, list[_Tag]]]:
    """Scan the main file and, recursively, everything it `<include>`s (relative to the including file, then main)."""
    scanned: dict[Path, tuple[bytes, list[_Tag]]] = {}

    def visit(path: Path) -> None:
        if path in scanned:
            return
        scanned[path] = _scan_file(path)
        for tag in scanned[path][1]:
            if tag.name == "include":
                visit(_include_target(tag, path, main_path))

    visit(main_path)
    return scanned


def _include_target(tag: _Tag, including: Path, main_path: Path) -> Path:
    name = tag.attributes.get("file")
    if not name:
        raise ExportError(f"<include> without 'file' in {including}")
    for base in (including.parent, main_path.parent):
        candidate = (base / name).resolve()
        if candidate.is_file():
            return candidate
    raise ExportError(f"<include file='{name}'> in {including} not found (looked next to it and next to {main_path})")


def _locate(
    scanned: Mapping[Path, tuple[bytes, list[_Tag]]], joint: str
) -> tuple[tuple[Path, _Tag], tuple[Path, _Tag]]:
    """The unique `<joint name=joint>` and `<position name=joint>` start tags across all files, or ExportError."""
    joints, actuators, other_actuators = [], [], []
    for path, (_, tags) in scanned.items():
        for tag in tags:
            if tag.attributes.get("name") != joint:
                continue
            if tag.name == "joint" and tag.parent not in _NOT_JOINT_PARENTS:
                joints.append((path, tag))
            elif tag.parent == "actuator":
                (actuators if tag.name == "position" else other_actuators).append((path, tag))
    if len(joints) != 1:
        raise ExportError(
            f"joint '{joint}' appears {len(joints)} times as an explicit <joint name='{joint}'> "
            "(0 means it is generated by a macro such as <replicate>/<frame>; >1 means ambiguous). Refusing to guess."
        )
    if other_actuators:
        raise ExportError(f"actuator '{joint}' is a <{other_actuators[0][1].name}>, not <position>: kp/kv do not apply")
    if len(actuators) != 1:
        raise ExportError(f"expected exactly one <position name='{joint}'> actuator, found {len(actuators)}")
    return joints[0], actuators[0]


def _output_name(path: Path, main_path: Path, main_output: Path) -> Path:
    return main_output if path == main_path else path.with_name(f"{path.stem}_calibrated{path.suffix}")


def _eligible_parameters(fit: JointFit) -> dict[str, float]:
    """Dynamics values that are safe to export: only parameters the data actually identified."""
    values = fit.dynamics.to_dict()
    return {name: values[name] for name in DYNAMICS_FIELDS if name not in fit.unidentifiable}


def export_calibrated(
    xml_path: str | Path,
    out_path: str | Path,
    fits: Sequence[JointFit],
    fit_config: FitConfig | None = None,
    force: bool = False,
    overwrite: bool = False,
) -> ExportResult:
    """Write a calibrated copy of `xml_path` to `out_path`, plus `<out stem>.calibration.json`.

    Only accepted fits are written unless `force`; only parameters the data identified are written (the rest keep their
    inherited/original value). `out_path` must be in the same directory as `xml_path` so that relative `meshdir` and
    `include` paths keep working without copying assets. Source files are never modified.
    """
    main_path, out_path = Path(xml_path).resolve(), Path(out_path).resolve()
    if not main_path.is_file():
        raise DataError(f"MJCF not found: {main_path}")
    if out_path.parent != main_path.parent:
        raise ExportError(
            f"output must be in the same directory as the source ({main_path.parent}) so relative asset/include paths "
            f"keep working; got {out_path.parent}"
        )
    if not fits:
        raise ExportError("no fitted joints to export")
    scanned = _resolve_includes(main_path)
    if out_path in scanned:
        raise ExportError(f"output {out_path} would overwrite a source file of the model")

    exported: dict[str, dict[str, float]] = {}
    outcomes: list[ExportedJoint] = []
    edits: dict[Path, list[tuple[int, int, str]]] = {path: [] for path in scanned}
    for fit in fits:
        parameters = _eligible_parameters(fit)
        if not fit.accepted and not force:
            outcomes.append(
                ExportedJoint(fit.joint, False, "fit not accepted (cost above noise-floor margin); use force")
            )
            continue
        if not parameters:
            outcomes.append(ExportedJoint(fit.joint, False, "no parameter was identified"))
            continue
        (joint_path, joint_tag), (actuator_path, actuator_tag) = _locate(scanned, fit.joint)
        _queue_edit(
            edits, scanned, joint_path, joint_tag, {k: v for k, v in parameters.items() if k in _JOINT_ATTRIBUTES}
        )
        _queue_edit(
            edits, scanned, actuator_path, actuator_tag, {k: v for k, v in parameters.items() if k in ("kp", "kv")}
        )
        exported[fit.joint] = parameters
        outcomes.append(ExportedJoint(fit.joint, True, "", parameters))
    if not exported:
        raise ExportError("nothing to export: " + "; ".join(f"{o.joint}: {o.reason}" for o in outcomes))

    outputs = _render_outputs(scanned, edits, main_path, out_path)
    calibration_path = out_path.with_name(f"{out_path.stem}.calibration.json")
    for path in (*outputs.keys(), calibration_path):
        if path.exists() and not overwrite:
            raise ExportError(f"{path} already exists; pass overwrite=True (or --overwrite) to replace it")
    for path, content in outputs.items():
        path.write_bytes(content)
    try:
        _verify(main_path, out_path, fits, exported)
    except ExportError:
        for path in outputs:
            path.unlink(missing_ok=True)
        raise
    calibration_path.write_text(
        json.dumps(_calibration_record(main_path, outputs, fits, outcomes, fit_config), indent=2)
    )
    log.info("wrote %s (+%d included file(s)) and %s", out_path.name, len(outputs) - 1, calibration_path.name)
    return ExportResult(tuple(outputs), calibration_path, tuple(outcomes))


def _queue_edit(
    edits: dict[Path, list[tuple[int, int, str]]],
    scanned: Mapping[Path, tuple[bytes, list[_Tag]]],
    path: Path,
    tag: _Tag,
    attributes: Mapping[str, float],
) -> None:
    if not attributes:
        return
    text = scanned[path][0][tag.start : tag.end].decode()
    for attribute, value in attributes.items():
        text = _set_attribute(text, attribute, value)
    edits[path].append((tag.start, tag.end, text))


def _render_outputs(
    scanned: Mapping[Path, tuple[bytes, list[_Tag]]],
    edits: Mapping[Path, list[tuple[int, int, str]]],
    main_path: Path,
    main_output: Path,
) -> dict[Path, bytes]:
    """Apply edits; re-emit every file that changed or includes a file that changed (include paths rewritten)."""
    changed = {path for path, file_edits in edits.items() if file_edits}
    grew = True
    while grew:  # propagate up the include chain
        grew = False
        for path, (_, tags) in scanned.items():
            if path not in changed and any(
                tag.name == "include" and _include_target(tag, path, main_path) in changed for tag in tags
            ):
                changed.add(path)
                grew = True
    rendered: dict[Path, bytes] = {}
    for path in sorted(changed, key=lambda p: p != main_path):  # main file first
        data, tags = scanned[path]
        file_edits = list(edits[path])
        for tag in tags:
            if tag.name == "include" and (target := _include_target(tag, path, main_path)) in changed:
                new_name = _output_name(target, main_path, main_output).name
                file_edits.append((tag.start, tag.end, _set_include(data[tag.start : tag.end].decode(), new_name)))
        for start, end, text in sorted(file_edits, reverse=True):
            data = data[:start] + text.encode() + data[end:]
        rendered[_output_name(path, main_path, main_output)] = data
    return rendered


def _set_include(tag_text: str, new_name: str) -> str:
    """Point an <include> at the re-emitted file, keeping any directory part of the original reference."""

    def replace(match: re.Match) -> str:
        directory = match.group(2)[1:-1].rpartition("/")[0]
        return f'file{match.group(1)}"{directory + "/" if directory else ""}{new_name}"'

    return re.sub(r"(?<=\s)file(\s*=\s*)(\"[^\"]*\"|'[^']*')", replace, tag_text, count=1)


def _verify(source: Path, output: Path, fits: Sequence[JointFit], exported: Mapping[str, Mapping[str, float]]) -> None:
    """Recompile the output: exported values read back, everything else unchanged. Raises ExportError otherwise."""
    try:
        before, after = load_spec(source, disable_contacts=False), load_spec(output, disable_contacts=False)
        compiled_before, compiled_after = before.copy().compile(), after.copy().compile()
    except (DataError, ValueError) as error:
        raise ExportError(f"the calibrated model does not compile: {error}") from error
    if (compiled_before.nq, compiled_before.nu, compiled_before.nbody) != (
        compiled_after.nq,
        compiled_after.nu,
        compiled_after.nbody,
    ):
        raise ExportError("the calibrated model has a different structure than the source")
    for fit in fits:
        expected = read_dynamics(before, fit.joint).to_dict()
        expected.update(exported.get(fit.joint, {}))
        actual = read_dynamics(after, fit.joint).to_dict()
        for name in DYNAMICS_FIELDS:
            if abs(actual[name] - expected[name]) > _READBACK_RELATIVE_TOLERANCE * max(abs(expected[name]), 1e-12):
                raise ExportError(
                    f"read-back mismatch for {fit.joint}.{name}: wrote {expected[name]}, model has {actual[name]}"
                )


def _calibration_record(
    source: Path,
    outputs: Mapping[Path, bytes],
    fits: Sequence[JointFit],
    outcomes: Sequence[ExportedJoint],
    config: FitConfig | None,
) -> dict:
    status = {outcome.joint: outcome for outcome in outcomes}
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "autosysid_version": __version__,
        "mujoco_version": mujoco.__version__,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_xml": source.name,  # names only: the sidecar stays valid when the folder is moved or shared
        "output_xml": [path.name for path in outputs],
        "fit_config": asdict(config) if config is not None else None,
        "joints": {
            fit.joint: {
                "exported": status[fit.joint].exported,
                "skipped_reason": status[fit.joint].reason,
                "written_to_mjcf": status[fit.joint].written,
                "nominal": fit.nominal.to_dict(),
                "fitted": fit.dynamics.to_dict(),
                "delay_s": fit.delay_s,  # not an MJCF quantity: command->measurement latency for your controller
                "relative_ci_95": fit.relative_ci,
                "unidentifiable": list(fit.unidentifiable),
                "accepted": fit.accepted,
                "cost": fit.cost,
                "noise_floor": fit.noise_floor,
                "cost_over_noise_floor": fit.cost / fit.noise_floor if fit.noise_floor > 0 else None,
                "attempt_costs": list(fit.attempt_costs),
            }
            for fit in fits
        },
    }
