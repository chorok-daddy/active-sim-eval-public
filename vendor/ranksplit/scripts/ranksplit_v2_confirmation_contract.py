"""Fail-closed contract for the independent RankSplit-v2 confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interstitial_scenario_contract import validate_manifest_structure


ROOT = Path(__file__).resolve().parents[1]
POLICY_ORDER = ("octo-small", "octo-base", "rt-1-x")
PROGRAM_ORDER = (
    "source-eig",
    "ranksplit-v1:lambda-0.25",
    "ranksplit-v2:lambda-0.25",
)
FROZEN_STATUS = "frozen-before-ranksplit-v2-policy-outcomes"
OPERATIONAL_REPAIR_STATUS = (
    "frozen-after-sealed-partial-replay-before-outcome-inspection"
)
SOURCE_RACE_RECEIPT_SHA256 = (
    "97703eeca9b477f54b42f25d4b1f4d6b151f2ab6bb4eed48a1673bd768f4c5c6"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rooted(value: str) -> Path:
    return ROOT / value


def load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    status = protocol.get("status")
    if status not in {FROZEN_STATUS, OPERATIONAL_REPAIR_STATUS}:
        raise ValueError("RankSplit-v2 protocol is not frozen")
    for name, record in protocol["sealed_inputs"].items():
        target = rooted(str(record["path"]))
        if not target.is_file() or sha256(target) != record["sha256"]:
            raise ValueError(f"sealed input hash mismatch: {name}")
    manifest_record = protocol["sealed_inputs"]["scenario_manifest"]
    manifest = json.loads(
        rooted(str(manifest_record["path"])).read_text(encoding="utf-8")
    )
    validate_manifest_structure(manifest)
    if manifest.get("manifest_sha256") != manifest_record["payload_sha256"]:
        raise ValueError("scenario manifest payload identity changed")
    design = protocol["scenario_design"]
    if (
        len(manifest["scenarios"]) != int(design["scenario_count"])
        or len({row["descriptor_sha256"] for row in manifest["scenarios"]})
        != int(design["required_unique_descriptors"])
        or int(manifest["geometry_gates"]["official_pose_overlap_count"]) != 0
        or int(manifest["geometry_gates"]["prior_manifest_pose_overlap_count"]) != 0
    ):
        raise ValueError("scenario block violates frozen disjointness")
    receipt_record = protocol["sealed_inputs"]["runtime_geometry_receipt"]
    receipt = json.loads(
        rooted(str(receipt_record["path"])).read_text(encoding="utf-8")
    )
    if (
        receipt.get("status") != "pass"
        or receipt.get("full_validation") is not True
        or receipt.get("policy_outcomes_read") is not False
        or int(receipt.get("validated_scenarios", -1))
        != int(design["scenario_count"])
        or receipt.get("manifest_file_sha256") != manifest_record["sha256"]
    ):
        raise ValueError("runtime geometry receipt is not outcome-blind PASS")
    if tuple(row["role"] for row in protocol["policies"]) != POLICY_ORDER:
        raise ValueError("policy order changed")
    if tuple(row["role"] for row in protocol["programs"]) != PROGRAM_ORDER:
        raise ValueError("program order changed")
    sampling = protocol["sampling_design"]
    if (
        list(map(int, sampling["budgets"])) != [48, 96, 192, 408, 816]
        or list(map(int, sampling["primary_budgets"])) != [48, 96, 192]
        or int(sampling["expected_initial_count"]) != 12
        or sampling["without_replacement"] is not True
    ):
        raise ValueError("sampling design changed")
    if int(protocol["replay"]["repetitions"]) != 200:
        raise ValueError("repetition count changed")
    if status == OPERATIONAL_REPAIR_STATUS:
        amendment = protocol.get("operational_amendment", {})
        repair = amendment.get("progress_publication_repair", {})
        if (
            amendment.get("classification") != "SOURCE_RACE_REPRODUCED"
            or amendment.get("source_race_receipt_sha256")
            != SOURCE_RACE_RECEIPT_SHA256
            or amendment.get("scientific_outcomes_inspected") is not False
            or amendment.get("scientific_design_changed") is not False
            or amendment.get("old_checkpoint_count") != 71
            or amendment.get("old_checkpoint_reuse") is not False
            or amendment.get("clean_recomputation_required") is not True
            or amendment.get("fresh_output_root_required") is not True
            or amendment.get("fresh_checkpoint_root_required") is not True
            or repair.get("scope") != "outcome-free-progress-publication-only"
            or repair.get("replace_attempts") != 41
            or float(repair.get("retry_delay_seconds", -1.0)) != 0.05
            or float(repair.get("maximum_retry_delay_seconds", -1.0)) != 2.0
        ):
            raise ValueError("RankSplit-v2 operational amendment changed")
    gate = protocol["gate"]
    if (
        gate["primary_metric"] != "task_weak_order_exact_rate"
        or float(gate["preference_lambda"]) != 0.25
        or float(gate["safeguard_margins"]["mean_decision_regret"]) != 0.005
        or float(gate["safeguard_margins"]["mean_task_gap_mae"]) != 0.01
    ):
        raise ValueError("RankSplit-v2 gate changed")
    return protocol


def validate_raw_payload(payload, *, role: str, protocol, manifest):
    expected = next(row for row in protocol["policies"] if row["role"] == role)
    expected_by_id = {str(row["scenario_id"]): row for row in manifest["scenarios"]}
    if payload.get("status") != "pass" or payload.get("policy_name") != role:
        raise ValueError(f"{role} raw output is not a complete PASS")
    expected_count = len(expected_by_id)
    for key in ("scenario_count", "total_jobs", "completed_jobs", "episodes"):
        if int(payload.get(key, -1)) != expected_count:
            raise ValueError(f"{role} raw count mismatch: {key}")
    manifest_hash = protocol["sealed_inputs"]["scenario_manifest"]["sha256"]
    if payload.get("scenario_manifest_sha256") != manifest_hash:
        raise ValueError(f"{role} raw manifest hash mismatch")
    if payload.get("policy_seed") != expected["policy_seed"]:
        raise ValueError(f"{role} policy seed mismatch")
    if payload.get("checkpoints") != [expected["checkpoint"]]:
        raise ValueError(f"{role} checkpoint mismatch")
    indexed = {}
    for row in payload.get("results", []):
        scenario_id = str(row.get("scenario_id"))
        if scenario_id in indexed or scenario_id not in expected_by_id:
            raise ValueError(f"{role} unexpected or duplicate scenario")
        descriptor = expected_by_id[scenario_id]
        if (
            row.get("env_id") != descriptor["env_id"]
            or row.get("scenario_descriptor_sha256")
            != descriptor["descriptor_sha256"]
            or row.get("scenario_manifest_sha256") != manifest_hash
            or row.get("checkpoint") != expected["checkpoint"]
            or row.get("policy_seed") != expected["policy_seed"]
            or not isinstance(row.get("final_success"), bool)
        ):
            raise ValueError(f"{role} row contract mismatch: {scenario_id}")
        indexed[scenario_id] = row
    if set(indexed) != set(expected_by_id):
        raise ValueError(f"{role} raw output does not cover frozen block")
    return indexed


def validate_raw_files(protocol_path: Path, raw_paths: dict[str, Path]):
    protocol = load_protocol(protocol_path)
    manifest = json.loads(
        rooted(protocol["sealed_inputs"]["scenario_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if tuple(raw_paths) != POLICY_ORDER:
        raise ValueError("raw path order differs from frozen policy order")
    indexed = {
        role: validate_raw_payload(
            json.loads(path.read_text(encoding="utf-8")),
            role=role,
            protocol=protocol,
            manifest=manifest,
        )
        for role, path in raw_paths.items()
    }
    return protocol, manifest, indexed
