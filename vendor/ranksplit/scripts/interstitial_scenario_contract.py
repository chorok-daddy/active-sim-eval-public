"""Pure contract helpers for disjoint ManiSkill3 interstitial scenarios.

The transforms are outcome-independent.  They move the official object poses
toward the center of their task-specific position support while preserving
height and quaternion.  The sink target remains fixed because it is a virtual
goal plane rather than the manipulated object.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


ENV_IDS = (
    "PutCarrotOnPlateInScene-v1",
    "PutSpoonOnTableClothInScene-v1",
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "PutEggplantInBasketScene-v1",
)

EXPECTED_OFFICIAL_COUNTS = {
    "PutCarrotOnPlateInScene-v1": 24,
    "PutSpoonOnTableClothInScene-v1": 24,
    "StackGreenCubeOnYellowCubeBakedTexInScene-v1": 24,
    "PutEggplantInBasketScene-v1": 64,
}

TRANSFORMS = (
    {"id": "support-inset-1over8", "scale": 0.875},
    {"id": "support-inset-1over4", "scale": 0.75},
)

SINK_ENV_ID = "PutEggplantInBasketScene-v1"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_value(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def movable_object_indices(env_id: str, object_count: int) -> tuple[int, ...]:
    if env_id not in ENV_IDS:
        raise ValueError(f"unsupported environment: {env_id}")
    if object_count < 2:
        raise ValueError("interstitial scenarios require source and target objects")
    # The sink target is a fixed virtual goal plane.  All other official
    # Bridge tasks place both source and target objects from the pose table.
    return (0,) if env_id == SINK_ENV_ID else tuple(range(object_count))


def transform_positions(
    xyz_configs: Any,
    *,
    env_id: str,
    position_index: int,
    scale: float,
) -> Any:
    import numpy as np

    xyz = np.asarray(xyz_configs, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("xyz_configs must have shape [position, object, xyz]")
    if not 0 <= position_index < len(xyz):
        raise IndexError("position_index is outside the official pose table")
    if not 0.0 < scale < 1.0:
        raise ValueError("interstitial scale must lie strictly inside (0, 1)")
    movable = movable_object_indices(env_id, xyz.shape[1])
    center = xyz[:, movable, :2].reshape(-1, 2).mean(axis=0)
    transformed = xyz[position_index].copy()
    transformed[list(movable), :2] = center + scale * (
        transformed[list(movable), :2] - center
    )
    return transformed


def descriptor(
    *,
    env_id: str,
    scenario_id: str,
    base_episode_id: int,
    position_index: int,
    orientation_index: int,
    transform_id: str,
    transform_scale: float,
    object_names: Sequence[str],
    positions: Sequence[Sequence[float]],
    quaternions: Sequence[Sequence[float]],
    source_target_xy_distance_ratio_to_base: float = 1.0,
) -> dict[str, object]:
    if len(object_names) != len(positions) or len(object_names) != len(quaternions):
        raise ValueError("object names, positions, and quaternions must align")
    objects = [
        {
            "name": str(name),
            "position": [float(value) for value in position],
            "quaternion_wxyz": [float(value) for value in quaternion],
        }
        for name, position, quaternion in zip(object_names, positions, quaternions)
    ]
    record: dict[str, object] = {
        "env_id": env_id,
        "scenario_id": scenario_id,
        "base_episode_id": int(base_episode_id),
        "position_index": int(position_index),
        "orientation_index": int(orientation_index),
        "transform_id": transform_id,
        "transform_scale": float(transform_scale),
        "source_target_xy_distance_ratio_to_base": float(
            source_target_xy_distance_ratio_to_base
        ),
        "objects": objects,
    }
    record["descriptor_sha256"] = sha256_value(record)
    return record


def validate_descriptor(record: dict[str, object]) -> None:
    supplied = str(record.get("descriptor_sha256", ""))
    projected = dict(record)
    projected.pop("descriptor_sha256", None)
    if supplied != sha256_value(projected):
        raise ValueError(f"descriptor hash mismatch: {record.get('scenario_id')}")
    objects = list(record.get("objects", []))
    if len(objects) < 2:
        raise ValueError("scenario descriptor must contain at least two objects")
    for obj in objects:
        position = list(obj.get("position", []))
        quaternion = list(obj.get("quaternion_wxyz", []))
        if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
            raise ValueError("object position must contain three finite values")
        if len(quaternion) != 4 or not all(math.isfinite(float(v)) for v in quaternion):
            raise ValueError("object quaternion must contain four finite values")
        norm = math.sqrt(sum(float(v) ** 2 for v in quaternion))
        if abs(norm - 1.0) > 2e-4:
            raise ValueError("object quaternion is not normalized")
    ratio = float(record.get("source_target_xy_distance_ratio_to_base", -1.0))
    if not 0.75 - 1e-9 <= ratio <= 1.05 + 1e-9:
        raise ValueError("source-target XY distance distortion exceeds frozen bounds")


def validate_manifest_structure(payload: dict[str, object]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported interstitial manifest schema")
    if payload.get("status") != "pre-result-geometry-validated":
        raise ValueError("interstitial manifest is not geometry validated")
    scenarios = list(payload.get("scenarios", []))
    expected_total = 2 * sum(EXPECTED_OFFICIAL_COUNTS.values())
    if len(scenarios) != expected_total:
        raise ValueError(f"expected {expected_total} scenarios, got {len(scenarios)}")
    ids = [str(row.get("scenario_id")) for row in scenarios]
    hashes = [str(row.get("descriptor_sha256")) for row in scenarios]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("scenario ids and descriptors must be unique")
    for row in scenarios:
        validate_descriptor(row)
    projected = dict(payload)
    supplied = str(projected.pop("manifest_sha256", ""))
    if supplied != sha256_value(projected):
        raise ValueError("manifest hash mismatch")


def load_manifest_record(
    path: Path, *, scenario_id: str, expected_env_id: str
) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest_structure(payload)
    matches = [
        row
        for row in payload["scenarios"]
        if str(row.get("scenario_id")) == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one scenario record for {scenario_id}")
    record = matches[0]
    if str(record.get("env_id")) != expected_env_id:
        raise ValueError("scenario environment does not match requested environment")
    return record, hashlib.sha256(path.read_bytes()).hexdigest()


def install_pose_table(base: object, record: dict[str, object]) -> None:
    """Replace one instantiated Bridge environment's pose tables with one row."""

    import torch

    validate_descriptor(record)
    objects = list(record["objects"])
    expected_names = list(base.objs.keys())
    supplied_names = [str(obj["name"]) for obj in objects]
    if supplied_names != expected_names:
        raise ValueError(
            f"scenario object order mismatch: {supplied_names} != {expected_names}"
        )
    positions = torch.tensor(
        [[obj["position"] for obj in objects]],
        dtype=torch.float32,
        device=base.device,
    )
    quaternions = torch.tensor(
        [[obj["quaternion_wxyz"] for obj in objects]],
        dtype=torch.float32,
        device=base.device,
    )
    if positions.shape != (1, len(objects), 3):
        raise ValueError("installed position table has the wrong shape")
    if quaternions.shape != (1, len(objects), 4):
        raise ValueError("installed quaternion table has the wrong shape")
    base.xyz_configs = positions
    base.quat_configs = quaternions
