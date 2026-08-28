"""Validate three new raw batches and build a tensor for custom-result replay.

Uses the original raw validators. It does not replace missing episodes with
zeros, load expected outcomes, or claim to verify fresh simulator geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import paper_reproduction as core


def build_tensor(study, acquisition_spec, raw_payloads, raw_hashes):
    if study not in ("mechanism", "ranking"):
        raise ValueError("unknown simulation study")
    entry = core.manifest()["studies"][study]
    spec = core.checked_json(entry["spec"])
    scenarios = core.checked_json(entry["scenarios"])
    if acquisition_spec.get("study") != study or acquisition_spec.get("scenario_manifest_sha256") != entry["scenarios"]["sha256"]:
        raise ValueError("acquisition specification is for a different scenario set")
    policies = acquisition_spec.get("policies", [])
    if [p.get("role") for p in policies] != spec["policy_order"] or list(raw_payloads) != spec["policy_order"] or list(raw_hashes) != spec["policy_order"]:
        raise ValueError("expected three policies in canonical order")
    for p in policies:
        if not isinstance(p.get("checkpoint"), str) or not p["checkpoint"]:
            raise ValueError("record the exact checkpoint before acquisition")
        expected_seed = None if p["role"] == "rt-1-x" else 0
        if p.get("policy_seed") != expected_seed:
            raise ValueError("policy seed differs from the published acquisition design")
    protocol = {"policies": policies, "sealed_inputs": {"scenario_manifest": {
        "sha256": entry["scenarios"]["sha256"]}}}
    contract = core.frozen_module("ranksplit_v2_confirmation_contract.py" if study == "mechanism"
                                  else "ranksplit_global_u1536_amended_comparator_contract.py")
    indexed = {role: contract.validate_raw_payload(payload, role=role, protocol=protocol, manifest=scenarios)
               for role, payload in raw_payloads.items()}
    records = []
    for descriptor in scenarios["scenarios"]:
        row = {k: descriptor[k] for k in ("env_id", "scenario_id", "transform_id", "descriptor_sha256")}
        row["policies"] = {}
        for role in spec["policy_order"]:
            result = indexed[role][descriptor["scenario_id"]]
            row["policies"][role] = {
                "outcome": int(result["final_success"]),
                "initial_observation_sha256": str(result["initial_observation_sha256"]),
                "action_sha256": str(result["environment_action_sha256"]),
            }
        records.append(row)
    tensor = {
        "schema_version": 1, "status": "pass", "analysis_type": "new-outcome custom tensor",
        "scenario_count": len(records), "policy_order": spec["policy_order"],
        "scenario_records": records,
        "acquisition": {"scenario_manifest_sha256": entry["scenarios"]["sha256"],
                        "spec": acquisition_spec, "spec_sha256": core.digest(core.encoded(acquisition_spec)),
                        "raw_sha256": raw_hashes,
                        "validation": "static original raw-payload validation; not fresh geometry verification"},
    }
    core.validate_tensor(tensor, spec, scenarios)
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("mechanism", "ranking"), required=True)
    parser.add_argument("--acquisition-spec", type=Path, required=True)
    for flag in ("small", "base", "rt1x"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"octo-small": args.small, "octo-base": args.base, "rt-1-x": args.rt1x}
    payloads = {role: path.read_bytes() for role, path in paths.items()}
    result = build_tensor(args.study, json.loads(args.acquisition_spec.read_text(encoding="utf-8")),
                          {r: json.loads(b) for r, b in payloads.items()},
                          {r: hashlib.sha256(b).hexdigest() for r, b in payloads.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "validated-new-outcomes", "cells": result["scenario_count"]*3,
                      "reference_comparison": "not_applicable", "output": str(args.output)}))


if __name__ == "__main__":
    main()
