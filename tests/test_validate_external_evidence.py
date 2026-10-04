import copy
import json
import subprocess

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scripts import validate_external_evidence as validator
from pureharness.evaluation.external_evidence import ExternalEvidenceReceipt


PACK = Path(__file__).resolve().parents[1] / "benchmarks/external"


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pack validation must not execute subprocesses")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def data():
    # Small committed summary only: never the user's jobs or raw artifacts.
    return json.loads((PACK / "receipts/terminal-bench-2.1_regex-log_83abbf4.json").read_text())


def write_pack(tmp_path, data, name="one_83abbf4.json"):
    root = tmp_path / "external"
    (root / "receipts").mkdir(parents=True, exist_ok=True)
    write_receipt(root, name, data)
    write_index(root, [name])
    return root


def write_receipt(root, name, data):
    path = root / "receipts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_index(root, names):
    rows = "\n".join(f"| fixture | [JSON](receipts/{name}) |" for name in names)
    (root / "README.md").write_text(
        "# Fixture pack\n\n" + validator.INDEX_START +
        "\n| Task | Receipt |\n|---|---|\n" + rows + "\n" +
        validator.INDEX_END + "\n", encoding="utf-8",
    )


def change(data, path, value):
    keys = path.split(".")
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def another(data):
    value = copy.deepcopy(data)
    value["external"].update(trial_id="other-trial", trial_name="other-name", job_name="other-job")
    value["runtime"]["run_id"] = "other-run"
    return value


def test_real_committed_pack_passes_and_round_trips(capsys):
    assert validator.main([]) == 0
    assert capsys.readouterr().out == "External evidence pack valid: 5 receipt(s)\n"
    for path in sorted((PACK / "receipts").glob("*.json")):
        receipt = ExternalEvidenceReceipt.from_dict(json.loads(path.read_text()))
        assert receipt.to_json() == path.read_text()
        with pytest.raises(FrozenInstanceError):
            receipt.runtime.step_count = 0


@pytest.mark.parametrize("path,value,expected", [
    ("schema_version", 2, "schema"),
    ("schema_version", True, "schema_version"),
    ("receipt_type", "other", "receipt type"),
    ("external.trial_id", "", "trial_id"),
    ("external.pureharness_revision", "abc", "full Git SHA"),
    ("runtime.tool_call_count", -1, "non-negative"),
    ("runtime.tool_call_count", True, "non-negative"),
    ("runtime.tool_execution_count", 12, "tool_execution_count"),
    ("runtime.tool_result_error_count", 12, "tool_result_error_count"),
    ("runtime.protocol_completed", False, "protocol_completed"),
    ("external.reward", float("nan"), "finite number"),
    ("context.maximum_history_tokens", 8001, "maximum_history_tokens"),
    ("context.final_history_tokens", 1823, "final_history_tokens"),
    ("context.compacted_context_count", 11, "compacted_context_count"),
    ("progress.unique_tool_actions", 12, "progress.unique_tool_actions"),
    ("coding.verification_exit_nonzero", 2, "verification outcomes"),
    ("oracle.exceptions", 1, "oracle.exceptions"),
    ("offline_evaluation.progress_gap.final.active_steps", 3, "active_steps"),
])
def test_invalid_fields_and_contradictions(tmp_path, data, path, value, expected):
    change(data, path, value)
    result = validator.validate_pack(write_pack(tmp_path, data))
    assert any(expected in error for error in result.errors)


@pytest.mark.parametrize("section,key", [("external", "trial_id"), ("runtime", "run_id"),
                                        ("oracle", "reward"), ("context", "final_history_tokens")])
def test_absent_required_or_nullable_field_is_not_defaulted(tmp_path, data, section, key):
    del data[section][key]
    errors = validator.validate_pack(write_pack(tmp_path, data)).errors
    assert any(f"{section}.{key}: required field missing" in error for error in errors)


def test_missing_top_level_section_and_unknown_fields_fail(tmp_path, data):
    del data["context"]
    assert "required field missing" in validator.validate_pack(write_pack(tmp_path, data)).errors[0]
    data["context"] = None
    assert "expected an object" in validator.validate_pack(write_pack(tmp_path, data)).errors[0]
    data["context"] = {"raw_output": "do not print this"}
    with pytest.raises(ValueError):
        ExternalEvidenceReceipt.from_dict(data)


@pytest.mark.parametrize("count,detected,delivered,expected", [
    (1, 5, 4, "precedes detection"), (0, 5, 6, "zero count"),
    (1, None, None, "requires detected/delivered"), (1, 5, None, "requires detected/delivered"),
])
def test_advisory_evidence(tmp_path, data, count, detected, delivered, expected):
    data["stagnation_advisory"] = {
        "advisory_count": count, "first_detected_step": detected, "first_delivered_step": delivered,
    }
    assert any(expected in e for e in validator.validate_pack(write_pack(tmp_path, data)).errors)


def test_legitimate_null_evidence_stays_null(tmp_path, data):
    for section in ("progress", "coding", "offline_evaluation", "oracle"):
        data[section] = None
    for key in data["context"]:
        if key != "context_build_count":
            data["context"][key] = None
    data["context"]["context_build_count"] = 0
    for key in ("dataset", "model", "model_provider", "task_ref", "task_checksum", "harbor_version", "job_id", "reward"):
        data["external"][key] = None
    data["unavailable_evidence"] = ["progress_snapshot", "coding_evidence_snapshot", "offline unavailable"]
    root = write_pack(tmp_path, data)
    assert validator.validate_pack(root).errors == ()
    parsed = ExternalEvidenceReceipt.from_dict(data)
    assert parsed.offline_evaluation is None and parsed.external.reward is None
    assert parsed.to_dict() == data


@pytest.mark.parametrize("reward,reason", [(0, "completed"), (1, "max_steps_exceeded")])
def test_reward_is_independent_and_unhealthy_oracle_is_representable(tmp_path, data, reward, reason):
    data["external"]["reward"] = reward
    data["runtime"].update(end_reason=reason, protocol_completed=reason == "completed")
    data["oracle"].update(reward=0, exceptions=1, exception_type="OracleError")
    assert validator.validate_pack(write_pack(tmp_path, data)).errors == ()


@pytest.mark.parametrize("identity", ["trial_id", "location", "run_id"])
def test_duplicate_identities(tmp_path, data, identity):
    root = write_pack(tmp_path, data)
    other = another(data)
    if identity == "trial_id":
        other["external"]["trial_id"] = data["external"]["trial_id"]
    elif identity == "location":
        for key in ("job_name", "trial_name"):
            other["external"][key] = data["external"][key]
    else:
        other["runtime"]["run_id"] = data["runtime"]["run_id"]
    write_receipt(root, "two.json", other)
    write_index(root, ["one_83abbf4.json", "two.json"])
    assert any("duplicate" in error for error in validator.validate_pack(root).errors)


def test_filename_sha_mismatch(tmp_path, data):
    assert any("filename SHA" in error for error in validator.validate_pack(
        write_pack(tmp_path, data, "wrong_abcdef0.json"),
    ).errors)


@pytest.mark.parametrize("field,value", [("job_id", "different-id"), ("trial_id", "different-id"),
                                       ("reward", 0), ("task_checksum", "different-checksum")])
def test_conflicting_shared_oracle(tmp_path, data, field, value):
    root = write_pack(tmp_path, data)
    other = another(data)
    other["oracle"][field] = value
    if field == "task_checksum":
        other["external"][field] = value  # Each pair is valid on its own.
    write_receipt(root, "two.json", other)
    write_index(root, ["one_83abbf4.json", "two.json"])
    assert any("conflicting shared Oracle" in error for error in validator.validate_pack(root).errors)


def test_shared_task_ref_cannot_name_different_tasks(tmp_path, data):
    root = write_pack(tmp_path, data)
    other = another(data)
    other["oracle"] = None
    other["external"]["task"] = "other-task"
    write_receipt(root, "two.json", other)
    write_index(root, ["one_83abbf4.json", "two.json"])
    assert any("conflicting shared task identity.task" in e for e in validator.validate_pack(root).errors)


@pytest.mark.parametrize("names,expected", [
    ([], "missing from index"), (["missing.json"], "nonexistent receipt link"),
    (["one_83abbf4.json", "one_83abbf4.json"], "duplicate receipt index row"),
    (["../escape.json"], "invalid receipt path"),
])
def test_readme_index_failures(tmp_path, data, names, expected):
    root = write_pack(tmp_path, data)
    write_index(root, names)
    assert any(expected in error for error in validator.validate_pack(root).errors)


def test_index_markers_required_and_outside_links_ignored(tmp_path, data):
    root = write_pack(tmp_path, data)
    readme = root / "README.md"
    readme.write_text(readme.read_text() + "\n[unrelated](receipts/missing.json)\n")
    assert validator.validate_pack(root).errors == ()
    readme.write_text(readme.read_text().replace(validator.INDEX_START, ""))
    assert any("start/end marker" in e for e in validator.validate_pack(root).errors)


def test_deterministic_all_file_errors_and_no_mutation(tmp_path, data, capsys):
    root = write_pack(tmp_path, data)
    write_receipt(root, "a.json", {"invalid": "private text must not enter diagnostics"})
    write_receipt(root, "z.json", {"invalid": 1})
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    outputs = []
    for _ in range(2):
        assert validator.main([str(root)]) == 1
        output = capsys.readouterr()
        assert output.out == ""
        outputs.append(output.err)
    assert outputs[0] == outputs[1]
    assert "private text" not in outputs[0]
    errors = validator.validate_pack(root).errors
    assert errors == tuple(sorted(errors))
    assert any("a.json" in e for e in errors) and any("z.json" in e for e in errors)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_no_jobs_dependency_and_ignore_non_json(tmp_path, data, monkeypatch):
    root = write_pack(tmp_path, data)
    (root / "receipts/.gitkeep").write_text("")
    (root / "receipts/notes.txt").write_text("not a receipt")
    original = Path.read_text
    def confined_read(path, *args, **kwargs):
        assert path.is_relative_to(root), "validation attempted an out-of-pack read"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", confined_read)
    assert validator.validate_pack(root).errors == ()
    assert not (tmp_path / "jobs").exists()


def test_bad_json_empty_pack_and_symlink_escape(tmp_path, data):
    root = write_pack(tmp_path, data)
    path = root / "receipts/one_83abbf4.json"
    path.write_text("not JSON")
    assert any("invalid JSON" in e for e in validator.validate_pack(root).errors)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(data))
    path.unlink()
    path.symlink_to(outside)
    assert any("in-pack" in e for e in validator.validate_pack(root).errors)
    path.unlink()
    assert any("no JSON receipts" in e for e in validator.validate_pack(root).errors)


def test_unknown_payload_is_not_silently_dropped(data):
    data["runtime"]["raw_output"] = "untrusted text"
    with pytest.raises(ValueError, match="unknown fields"):
        ExternalEvidenceReceipt.from_dict(data)


def test_duplicate_json_fields_are_not_silently_overwritten(tmp_path, data):
    root = write_pack(tmp_path, data)
    path = root / "receipts/one_83abbf4.json"
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 2, "schema_version": 1'))
    assert any("duplicate field names" in e for e in validator.validate_pack(root).errors)


def test_partial_optional_progress_still_checks_known_counts(tmp_path, data):
    data["progress"].update(unique_tool_actions=12, repeated_tool_actions=None)
    assert any("progress.unique_tool_actions" in e for e in validator.validate_pack(write_pack(tmp_path, data)).errors)


def test_different_trials_and_revisions_can_have_different_metrics(tmp_path, data):
    root = write_pack(tmp_path, data)
    other = another(data)
    other["external"]["pureharness_revision"] = "a" * 40
    other["runtime"].update(step_count=20, model_call_count=20, tool_call_count=12)
    write_receipt(root, "two.json", other)
    write_index(root, ["one_83abbf4.json", "two.json"])
    assert validator.validate_pack(root).errors == ()


def test_recursive_discovery_and_stable_success_output(tmp_path, data, capsys):
    root = write_pack(tmp_path, data, "nested/one.json")
    outputs = []
    for _ in range(2):
        assert validator.main([str(root)]) == 0
        outputs.append(capsys.readouterr())
    assert outputs[0] == outputs[1]
    assert outputs[0].out == "External evidence pack valid: 1 receipt(s)\n"
    assert outputs[0].err == ""


def test_shared_null_fact_does_not_hide_conflict_or_misidentify_its_source(tmp_path, data):
    data["oracle"]["reward"] = None
    root = write_pack(tmp_path, data, "a.json")
    second = another(data)
    second["oracle"]["reward"] = 1
    third = another(data)
    third["external"].update(trial_id="third-id", trial_name="third-name", job_name="third-job")
    third["runtime"]["run_id"] = "third-run"
    third["oracle"]["reward"] = 0
    write_receipt(root, "b.json", second)
    write_receipt(root, "c.json", third)
    write_index(root, ["a.json", "b.json", "c.json"])
    errors = validator.validate_pack(root).errors
    assert any("shared Oracle.reward; also in receipts/b.json" in e for e in errors)
    assert not any("shared Oracle.reward; also in receipts/a.json" in e for e in errors)
