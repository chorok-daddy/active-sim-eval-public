"""Fail-closed contract for the global-U=1536 amended comparator study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from interstitial_scenario_contract import validate_manifest_structure


ROOT = Path(__file__).resolve().parents[1]
POLICY_ORDER = ("octo-small", "octo-base", "rt-1-x")
PROGRAM_ORDER = (
    "matched-eig",
    "saad-taskwise-d1:global-u1536",
    "srank-singleton",
    "ranksplit-v2:lambda-0.50",
    "ranksplit-v2:lambda-0.75",
)
FROZEN_STATUS = "frozen-before-global-u1536-amended-outcomes"
GLOBAL_UNION_EVENTS = 1536


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rooted(value: str) -> Path:
    return ROOT / value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(protocol.get("status") == FROZEN_STATUS, "amended comparator protocol is not frozen")
    for name, record in protocol["sealed_inputs"].items():
        target = _rooted(str(record["path"]))
        _require(target.is_file(), f"sealed input is missing: {name}")
        _require(sha256(target) == record["sha256"], f"sealed input hash mismatch: {name}")

    manifest_record = protocol["sealed_inputs"]["scenario_manifest"]
    manifest = json.loads(_rooted(str(manifest_record["path"])).read_text(encoding="utf-8"))
    validate_manifest_structure(manifest)
    _require(
        manifest.get("manifest_sha256") == manifest_record["payload_sha256"],
        "scenario manifest payload identity changed",
    )
    design = protocol["scenario_design"]
    _require(
        int(design["scenario_count"]) == 272
        and int(design["population_size"]) == 816
        and int(design["task_count"]) == 4
        and int(design["policy_count"]) == 3
        and len(manifest["scenarios"]) == int(design["scenario_count"])
        and len({row["descriptor_sha256"] for row in manifest["scenarios"]})
        == int(design["required_unique_descriptors"])
        and int(manifest["geometry_gates"]["official_pose_overlap_count"]) == 0
        and int(manifest["geometry_gates"]["prior_manifest_pose_overlap_count"]) == 0,
        "amended scenario block violates frozen disjointness design",
    )
    _require(
        tuple(row["id"] for row in manifest["construction"]["transforms"])
        == tuple(design["transform_ids"]),
        "amended scenario transform order changed",
    )
    _require(
        manifest["construction"].get("global_union_events") == GLOBAL_UNION_EVENTS
        and manifest["construction"].get("outcome_values_used") is False,
        "amended manifest global-U or outcome-blind marker changed",
    )

    receipt_record = protocol["sealed_inputs"]["runtime_geometry_receipt"]
    receipt = json.loads(_rooted(str(receipt_record["path"])).read_text(encoding="utf-8"))
    _require(
        receipt.get("status") == "pass"
        and receipt.get("full_validation") is True
        and receipt.get("policy_outcomes_read") is False
        and int(receipt.get("validated_scenarios", -1)) == int(design["scenario_count"])
        and receipt.get("manifest_file_sha256") == manifest_record["sha256"],
        "amended runtime geometry receipt is not a full outcome-blind PASS",
    )

    _require(
        tuple(row["role"] for row in protocol["policies"]) == POLICY_ORDER,
        "policy order changed",
    )
    _require(
        tuple(row["role"] for row in protocol["programs"]) == PROGRAM_ORDER,
        "amended program order changed",
    )
    expected_families = {
        "matched-eig": ("matched-eig", 0.0),
        "saad-taskwise-d1:global-u1536": ("saad-taskwise-d1-global-u1536", 0.0),
        "srank-singleton": ("srank-singleton", 0.0),
        "ranksplit-v2:lambda-0.50": ("ranksplit-v2", 0.5),
        "ranksplit-v2:lambda-0.75": ("ranksplit-v2", 0.75),
    }
    for row in protocol["programs"]:
        role = str(row["role"])
        family, preference_lambda = expected_families[role]
        _require(row["family"] == family, f"program family changed: {role}")
        _require(
            float(row["preference_lambda"]) == preference_lambda,
            f"program lambda changed: {role}",
        )

    sampling = protocol["sampling_design"]
    _require(
        list(map(int, sampling["budgets"])) == [48, 96, 192, 408, 816]
        and list(map(int, sampling["primary_budgets"])) == [48, 96, 192]
        and int(sampling["expected_initial_count"]) == 12
        and sampling["without_replacement"] is True
        and sampling["one_cell_cost"] == 1
        and sampling["shared_scenario_permutation_scope"] == "task-policy",
        "amended sampling design changed",
    )
    _require(
        sampling["outer_task_scheduler"]["rule"] == "minimum acquired task cells"
        and sampling["outer_task_scheduler"]["tie_break"]
        == "stable_seed(master, outer-task-tie, repetition, global_step, task)",
        "amended outer task scheduler changed",
    )

    replay = protocol["replay"]
    _require(
        int(replay["repetitions"]) == 200
        and int(replay["simulation_seed"]) == 20260810
        and list(map(int, replay["repetition_indices"])) == list(range(200)),
        "amended replay identity changed",
    )
    saad = protocol["allocators"]["saad_taskwise_d1_global_u1536"]
    _require(
        float(saad["delta"]) == 0.05
        and int(saad["global_union_events"]) == GLOBAL_UNION_EVENTS
        and saad["confidence_radius"]
        == "sqrt(log(2*global_union_events/delta)/(2*count))"
        and saad["pair_rule"] == "less-sampled active pair; stable hash on equal counts"
        and saad["true_tie_rule"] == "remain unresolved; never force strict order"
        and saad["shared_census_included"] is True,
        "global-U=1536 Saad adapter contract changed",
    )
    srank = protocol["allocators"]["srank_singleton"]
    _require(
        int(srank["horizon_per_task"]) == 204
        and int(srank["rounds"]) == 2
        and list(map(int, srank["round_targets"])) == [0, 34, 58]
        and list(map(int, srank["singleton_clusters"])) == [1, 1, 1],
        "SRank adapter contract changed",
    )

    gate = protocol["gate"]
    _require(
        gate["primary_program"] == "ranksplit-v2:lambda-0.50"
        and tuple(gate["primary_comparators"])
        == ("saad-taskwise-d1:global-u1536", "srank-singleton")
        and gate["multiple_comparison"]["method"]
        == "Bonferroni one-sided familywise alpha 0.05 over two primary gains"
        and gate["primary_metric"] == "task_weak_order_exact_rate"
        and float(gate["safeguard_margins"]["mean_task_gap_mae"]) == 0.01
        and float(gate["safeguard_margins"]["mean_decision_regret"]) == 0.005
        and gate["full_endpoint_identity_required"] is True,
        "amended comparator gate changed",
    )
    _require(protocol.get("native_reference_available") is False, "native reference status changed")
    return protocol


def validate_raw_payload(
    payload: dict[str, object], *, role: str, protocol: dict[str, object], manifest: dict[str, object]
) -> dict[str, dict[str, object]]:
    expected = next(row for row in protocol["policies"] if row["role"] == role)
    expected_by_id = {str(row["scenario_id"]): row for row in manifest["scenarios"]}
    _require(payload.get("status") == "pass", f"{role} raw output is not a complete PASS")
    _require(payload.get("policy_name") == role, f"{role} raw policy identity changed")
    expected_count = len(expected_by_id)
    for key in ("scenario_count", "total_jobs", "completed_jobs", "episodes"):
        _require(int(payload.get(key, -1)) == expected_count, f"{role} raw count mismatch: {key}")
    manifest_hash = protocol["sealed_inputs"]["scenario_manifest"]["sha256"]
    _require(payload.get("scenario_manifest_sha256") == manifest_hash, f"{role} raw manifest hash mismatch")
    _require(payload.get("policy_seed") == expected["policy_seed"], f"{role} policy seed mismatch")
    _require(payload.get("checkpoints") == [expected["checkpoint"]], f"{role} checkpoint mismatch")

    indexed: dict[str, dict[str, object]] = {}
    for row in payload.get("results", []):
        scenario_id = str(row.get("scenario_id"))
        _require(
            scenario_id not in indexed and scenario_id in expected_by_id,
            f"{role} unexpected or duplicate scenario",
        )
        descriptor = expected_by_id[scenario_id]
        _require(
            row.get("env_id") == descriptor["env_id"]
            and row.get("scenario_descriptor_sha256") == descriptor["descriptor_sha256"]
            and row.get("scenario_manifest_sha256") == manifest_hash
            and row.get("checkpoint") == expected["checkpoint"]
            and row.get("policy_seed") == expected["policy_seed"]
            and isinstance(row.get("final_success"), bool),
            f"{role} row contract mismatch: {scenario_id}",
        )
        indexed[scenario_id] = row
    _require(set(indexed) == set(expected_by_id), f"{role} raw output does not cover the frozen block")
    return indexed


def validate_raw_files(protocol_path: Path, raw_paths: dict[str, Path]):
    protocol = load_protocol(protocol_path)
    manifest = json.loads(
        _rooted(protocol["sealed_inputs"]["scenario_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    _require(tuple(raw_paths) == POLICY_ORDER, "raw path order differs from frozen policy order")
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


__all__ = [
    "FROZEN_STATUS",
    "GLOBAL_UNION_EVENTS",
    "POLICY_ORDER",
    "PROGRAM_ORDER",
    "load_protocol",
    "sha256",
    "validate_raw_files",
]
