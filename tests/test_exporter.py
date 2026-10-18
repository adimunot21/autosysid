import json
import os
import shutil
from pathlib import Path

import pytest

from autosysid.config import FitConfig
from autosysid.errors import ExportError
from autosysid.exporter import _set_attribute, export_calibrated
from autosysid.mock import MINI_ARM_XML
from autosysid.model import JointDynamics, load_spec, read_dynamics
from autosysid.optimizer import JointFit

FITTED = JointDynamics(kp=700.0, kv=4.1, damping=0.84, frictionloss=0.078, armature=0.0224)
SO101_DIR = Path(os.environ.get("AUTOSYSID_SO101_DIR", Path.home() / "dev/simtoreal/assets/so101"))


def make_fit(joint="shoulder_pan", dynamics=FITTED, accepted=True, unidentifiable=()) -> JointFit:
    nominal = read_dynamics(load_spec(MINI_ARM_XML), joint)
    return JointFit(
        joint=joint,
        nominal=nominal,
        dynamics=dynamics,
        delay_s=0.026,
        relative_ci={"kp": 0.1, "kv": 0.1, "damping": 0.1, "frictionloss": 0.4, "armature": 0.1, "delay": 0.0003},
        unidentifiable=tuple(unidentifiable),
        cost=2.4e-5,
        noise_floor=1.1e-5,
        accepted=accepted,
        attempt_costs=(2.4e-5, 3.0e-4, 2.5e-5),
        seconds=1.0,
    )


@pytest.fixture
def mini(tmp_path) -> Path:
    path = tmp_path / "arm.xml"
    shutil.copy(MINI_ARM_XML, path)
    return path


def changed_lines(before: Path, after: Path) -> list[int]:
    old, new = before.read_text().splitlines(), after.read_text().splitlines()
    assert len(old) == len(new)
    return [i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b]


def test_set_attribute_replaces_in_place_and_keeps_everything_else():
    tag = '<joint name="a"  damping = "0.6" class=\'x\' />'
    assert _set_attribute(tag, "damping", 0.84) == '<joint name="a"  damping = "0.84" class=\'x\' />'


def test_set_attribute_appends_before_closer_keeping_style():
    assert _set_attribute('<position name="a" />', "kp", 700.0) == '<position name="a" kp="700" />'
    assert _set_attribute('<position name="a"/>', "kp", 700.0) == '<position name="a" kp="700"/>'
    assert _set_attribute('<joint name="a">', "armature", 0.02) == '<joint name="a" armature="0.02">'


def test_set_attribute_does_not_match_a_longer_attribute_name():
    assert (
        _set_attribute('<joint name="a" xdamping="9"/>', "damping", 1.0) == '<joint name="a" xdamping="9" damping="1"/>'
    )


def test_set_attribute_keeps_higher_damping_coefficients():
    assert _set_attribute('<joint damping="0.6 0.1 0"/>', "damping", 0.9) == '<joint damping="0.9 0.1 0"/>'


def test_export_writes_explicit_values_and_reads_them_back(mini):
    out = mini.with_name("arm_calibrated.xml")
    result = export_calibrated(mini, out, [make_fit()], FitConfig())
    assert read_dynamics(load_spec(out), "shoulder_pan").to_dict() == pytest.approx(FITTED.to_dict(), rel=1e-6)
    assert result.xml_paths == (out.resolve(),) and result.joints[0].exported


def test_export_leaves_other_joints_and_the_class_defaults_untouched(mini):
    out = mini.with_name("arm_calibrated.xml")
    export_calibrated(mini, out, [make_fit()])
    original = load_spec(MINI_ARM_XML)
    calibrated = load_spec(out)
    for joint in ("shoulder_lift", "elbow_flex"):
        assert read_dynamics(calibrated, joint) == read_dynamics(original, joint)
    assert 'class="sts3215"' in out.read_text() and 'damping="0.60"' in out.read_text()  # class definition intact


def test_export_diff_is_exactly_the_two_start_tags(mini):
    out = mini.with_name("arm_calibrated.xml")
    export_calibrated(mini, out, [make_fit()])
    lines = changed_lines(mini, out)
    assert len(lines) == 2
    text = out.read_text().splitlines()
    assert all("shoulder_pan" in text[i] for i in lines)


def test_export_never_modifies_the_source(mini):
    before = mini.read_bytes()
    export_calibrated(mini, mini.with_name("arm_calibrated.xml"), [make_fit()])
    assert mini.read_bytes() == before


def test_unidentified_parameters_are_not_written(mini):
    out = mini.with_name("arm_calibrated.xml")
    export_calibrated(mini, out, [make_fit(unidentifiable=("frictionloss", "armature"))])
    pan = [line for line in out.read_text().splitlines() if 'name="shoulder_pan"' in line and "<joint" in line][0]
    assert "frictionloss" not in pan and "armature" not in pan and 'damping="0.84"' in pan
    nominal = read_dynamics(load_spec(MINI_ARM_XML), "shoulder_pan")
    fitted = read_dynamics(load_spec(out), "shoulder_pan")
    assert fitted.frictionloss == pytest.approx(nominal.frictionloss) and fitted.damping == pytest.approx(0.84)


def test_rejected_fit_is_skipped_unless_forced(mini):
    out = mini.with_name("arm_calibrated.xml")
    with pytest.raises(ExportError, match="nothing to export"):
        export_calibrated(mini, out, [make_fit(accepted=False)])
    assert not out.exists()
    result = export_calibrated(mini, out, [make_fit(accepted=False)], force=True)
    assert result.joints[0].exported


def test_mixed_batch_exports_accepted_and_reports_skipped(mini):
    fits = [make_fit(), make_fit("elbow_flex", accepted=False)]
    result = export_calibrated(mini, mini.with_name("arm_calibrated.xml"), fits)
    assert [(j.joint, j.exported) for j in result.joints] == [("shoulder_pan", True), ("elbow_flex", False)]
    record = json.loads(result.calibration_path.read_text())
    assert record["joints"]["elbow_flex"]["skipped_reason"] and not record["joints"]["elbow_flex"]["exported"]


def test_calibration_json_records_delay_cis_and_costs(mini):
    result = export_calibrated(mini, mini.with_name("arm_calibrated.xml"), [make_fit()], FitConfig())
    record = json.loads(result.calibration_path.read_text())
    joint = record["joints"]["shoulder_pan"]
    assert record["schema_version"] == 1 and record["fit_config"]["max_restarts"] == 2
    assert joint["delay_s"] == pytest.approx(0.026) and joint["relative_ci_95"]["kp"] == 0.1
    assert joint["cost_over_noise_floor"] == pytest.approx(2.4e-5 / 1.1e-5)
    assert set(joint["written_to_mjcf"]) == set(FITTED.to_dict())


def test_included_file_is_patched_and_the_include_rewritten(tmp_path):
    arm = tmp_path / "arm.xml"
    shutil.copy(MINI_ARM_XML, arm)
    main = tmp_path / "scene.xml"
    main.write_text('<mujoco model="scene">\n  <include file="arm.xml"/>\n</mujoco>\n')
    out = tmp_path / "scene_calibrated.xml"
    result = export_calibrated(main, out, [make_fit()])
    assert {p.name for p in result.xml_paths} == {"scene_calibrated.xml", "arm_calibrated.xml"}
    assert 'file="arm_calibrated.xml"' in out.read_text()
    assert read_dynamics(load_spec(out), "shoulder_pan").to_dict() == pytest.approx(FITTED.to_dict(), rel=1e-6)
    assert arm.read_bytes() == Path(MINI_ARM_XML).read_bytes()  # original include untouched


def test_nested_include_chain_and_relative_subdirectory(tmp_path):
    (tmp_path / "parts").mkdir()
    shutil.copy(MINI_ARM_XML, tmp_path / "parts" / "arm.xml")
    (tmp_path / "parts" / "mid.xml").write_text('<mujoco><include file="arm.xml"/></mujoco>')
    main = tmp_path / "scene.xml"
    main.write_text('<mujoco model="scene"><include file="parts/mid.xml"/></mujoco>')
    out = tmp_path / "scene_calibrated.xml"
    result = export_calibrated(main, out, [make_fit()])
    assert len(result.xml_paths) == 3
    assert read_dynamics(load_spec(out), "shoulder_pan").kp == pytest.approx(FITTED.kp, rel=1e-6)


def test_output_in_another_directory_is_refused(mini, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(ExportError, match="same directory"):
        export_calibrated(mini, other / "arm.xml", [make_fit()])


def test_overwriting_the_source_or_an_existing_output_is_refused(mini):
    with pytest.raises(ExportError, match="source file"):
        export_calibrated(mini, mini, [make_fit()], overwrite=True)
    out = mini.with_name("arm_calibrated.xml")
    export_calibrated(mini, out, [make_fit()])
    with pytest.raises(ExportError, match="already exists"):
        export_calibrated(mini, out, [make_fit()])
    export_calibrated(mini, out, [make_fit()], overwrite=True)


def test_non_position_actuator_is_refused(tmp_path):
    xml = (
        Path(MINI_ARM_XML)
        .read_text()
        .replace(
            '<position class="sts3215" name="shoulder_pan" joint="shoulder_pan" ctrlrange="-1.9 1.9"/>',
            '<motor name="shoulder_pan" joint="shoulder_pan"/>',
        )
    )
    path = tmp_path / "arm.xml"
    path.write_text(xml)
    with pytest.raises(ExportError, match="not <position>"):
        export_calibrated(path, tmp_path / "out.xml", [make_fit()])


def test_duplicate_or_missing_joint_is_refused(mini):
    ghost = make_fit()
    ghost = JointFit(**{**ghost.__dict__, "joint": "ghost"})
    with pytest.raises(ExportError, match="0 times"):
        export_calibrated(mini, mini.with_name("out.xml"), [ghost])
    duplicated = mini.read_text().replace("</worldbody>", '<body><joint name="shoulder_pan"/></body></worldbody>')
    mini.write_text(duplicated)
    with pytest.raises(ExportError, match="2 times"):
        export_calibrated(mini, mini.with_name("out.xml"), [make_fit()])


def test_failed_verification_leaves_no_output(mini, monkeypatch):
    import autosysid.exporter as exporter

    monkeypatch.setattr(exporter, "_format_number", lambda value: "1.0")  # wrong value written on purpose
    out = mini.with_name("arm_calibrated.xml")
    with pytest.raises(ExportError, match="read-back mismatch"):
        export_calibrated(mini, out, [make_fit()])
    assert not out.exists()


@pytest.mark.skipif(not (SO101_DIR / "scene.xml").is_file(), reason="SO-ARM100 SO101 model not found")
def test_real_so101_model_with_class_inheritance_and_include(tmp_path):
    """The real model: values inherited from <default class='sts3215'>, joints in an included file, 16 MB of meshes."""
    for name in ("scene.xml", "so101_new_calib.xml"):
        shutil.copy(SO101_DIR / name, tmp_path / name)
    (tmp_path / "assets").symlink_to(SO101_DIR / "assets", target_is_directory=True)  # read-only link, no copy
    nominal = read_dynamics(load_spec(tmp_path / "scene.xml"), "elbow_flex")
    fitted = JointDynamics(kp=640.0, kv=3.2, damping=0.7, frictionloss=0.06, armature=0.03)
    fit = JointFit(**{**make_fit("elbow_flex").__dict__, "nominal": nominal, "dynamics": fitted})
    out = tmp_path / "scene_calibrated.xml"
    export_calibrated(tmp_path / "scene.xml", out, [fit])
    calibrated = load_spec(out)
    assert read_dynamics(calibrated, "elbow_flex").to_dict() == pytest.approx(fitted.to_dict(), rel=1e-6)
    assert read_dynamics(calibrated, "shoulder_pan") == read_dynamics(load_spec(tmp_path / "scene.xml"), "shoulder_pan")
    assert changed_lines(tmp_path / "so101_new_calib.xml", tmp_path / "so101_new_calib_calibrated.xml") != []
    assert len(changed_lines(tmp_path / "so101_new_calib.xml", tmp_path / "so101_new_calib_calibrated.xml")) == 2
