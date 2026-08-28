"""Run one official-protocol WidowX episode using the WSL Octo server."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import socket
import statistics
import struct
import time

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import (  # noqa: F401
    PutCarrotOnPlateInScene,
    PutEggplantInBasketScene,
    PutSpoonOnTableClothInScene,
    StackGreenCubeOnYellowCubeBakedTexInScene,
)
from mani_skill.utils.geometry import rotation_conversions
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill3_obs_dict,
)

from interstitial_scenario_contract import install_pose_table, load_manifest_record


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("policy server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def request(
    connection: socket.socket,
    metadata: dict[str, object],
    payload: bytes = b"",
) -> dict[str, object]:
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    connection.sendall(
        struct.pack("!IQ", len(metadata_bytes), len(payload))
        + metadata_bytes
        + payload
    )
    response_size = struct.unpack("!I", recv_exact(connection, 4))[0]
    return json.loads(recv_exact(connection, response_size))


def to_environment_action(
    raw_action: list[list[float]], action_format: str
) -> np.ndarray:
    raw = torch.tensor(raw_action, dtype=torch.float32)
    if action_format == "axis_angle_binary_gripper":
        if raw.shape != (1, 7):
            raise ValueError(f"expected one 7D axis-angle action, got {raw.shape}")
        return raw[0].numpy()
    if action_format != "euler_gripper_probability":
        raise ValueError(f"unknown action format: {action_format}")
    world_vector = raw[:, :3]
    euler_delta = raw[:, 3:6]
    rotation = rotation_conversions.matrix_to_axis_angle(
        rotation_conversions.euler_angles_to_matrix(euler_delta, "XYZ")
    )
    gripper = 2.0 * (raw[:, 6:7] > 0.5).float() - 1.0
    return torch.cat((world_vector, rotation, gripper), dim=1)[0].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--socket-timeout-seconds",
        type=float,
        default=600.0,
        help="Fail-closed transport timeout; includes first-inference compilation.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=int)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--env-id", default="PutCarrotOnPlateInScene-v1")
    parser.add_argument("--scenario-manifest", type=Path)
    parser.add_argument("--scenario-id")
    parser.add_argument("--shutdown-server", action="store_true")
    args = parser.parse_args()
    episode_id = args.seed if args.episode_id is None else args.episode_id
    policy_seed = args.seed if args.policy_seed is None else args.policy_seed

    # Seed every host-side generator before constructing the environment.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="physx_cpu",
        render_backend="sapien_cpu",
    )

    try:
        scenario_record = None
        scenario_manifest_sha256 = None
        if (args.scenario_manifest is None) != (args.scenario_id is None):
            raise ValueError(
                "--scenario-manifest and --scenario-id must be supplied together"
            )
        reset_episode_id = episode_id
        if args.scenario_manifest is not None:
            scenario_record, scenario_manifest_sha256 = load_manifest_record(
                args.scenario_manifest,
                scenario_id=str(args.scenario_id),
                expected_env_id=args.env_id,
            )
            install_pose_table(env.unwrapped, scenario_record)
            reset_episode_id = 0
        obs, _ = env.reset(
            seed=args.seed,
            options={"episode_id": torch.tensor([reset_episode_id])},
        )
        instruction = str(env.unwrapped.get_language_instruction()[0])
        environment_seconds: list[float] = []
        roundtrip_seconds: list[float] = []
        preprocessing_seconds: list[float] = []
        inference_seconds: list[float] = []
        postprocessing_seconds: list[float] = []
        rewards: list[float] = []
        action_hasher = hashlib.sha256()
        observation_hasher = hashlib.sha256()
        initial_observation_sha256 = ""
        first_environment_action: list[float] = []
        terminated_value = False
        terminated_ever = False
        policy_terminated_ever = False
        truncated_value = False
        final_success = False
        success_ever = False

        if args.socket_timeout_seconds <= 0:
            raise ValueError("--socket-timeout-seconds must be positive")
        with socket.create_connection(
            (args.host, args.port), timeout=args.socket_timeout_seconds
        ) as connection:
            reset_response = request(
                connection,
                {
                    "type": "reset",
                    "instruction": instruction,
                    "policy_seed": policy_seed,
                },
            )
            episode_started = time.perf_counter()
            for step_index in range(1, 1001):
                image = get_image_from_maniskill3_obs_dict(env, obs)
                image_array = image.detach().cpu().numpy().astype(np.uint8, copy=False)
                image_bytes = image_array.tobytes(order="C")
                if step_index == 1:
                    initial_observation_sha256 = hashlib.sha256(
                        image_bytes
                    ).hexdigest()
                observation_hasher.update(image_bytes)
                roundtrip_started = time.perf_counter()
                response = request(
                    connection,
                    {"type": "step", "shape": list(image_array.shape)},
                    image_bytes,
                )
                roundtrip_seconds.append(time.perf_counter() - roundtrip_started)
                preprocessing_seconds.append(float(response["preprocess_seconds"]))
                inference_seconds.append(float(response["inference_seconds"]))
                postprocessing_seconds.append(float(response["postprocess_seconds"]))
                action = to_environment_action(
                    response["action"], str(response["action_format"])
                )
                policy_terminated_ever = policy_terminated_ever or bool(
                    response.get("predicted_terminate", False)
                )
                if step_index == 1:
                    first_environment_action = action.tolist()
                action_hasher.update(
                    np.asarray(action, dtype=np.float32).tobytes(order="C")
                )

                environment_started = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                environment_seconds.append(time.perf_counter() - environment_started)
                rewards.append(float(reward[0].item()))
                terminated_value = bool(terminated[0].item())
                terminated_ever = terminated_ever or terminated_value
                truncated_value = bool(truncated[0].item())
                final_success = bool(info["success"][0].item())
                success_ever = success_ever or final_success
                # Match SIMPLER's official ManiSkill3 evaluator: environment
                # `terminated` is recorded but does not end the rollout. Octo
                # emits no predicted terminate action, so only the time limit
                # (`truncated`) ends this episode.
                if truncated_value:
                    break

            episode_seconds = time.perf_counter() - episode_started
            if args.shutdown_server:
                request(connection, {"type": "shutdown"})

        report = {
            "probe_type": "official-protocol policy-conditioned baseline",
            "env_id": args.env_id,
            "instruction": instruction,
            "seed": args.seed,
            "episode_id": episode_id,
            "scenario_id": (
                None if scenario_record is None else scenario_record["scenario_id"]
            ),
            "scenario_descriptor_sha256": (
                None
                if scenario_record is None
                else scenario_record["descriptor_sha256"]
            ),
            "scenario_manifest_sha256": scenario_manifest_sha256,
            "scenario_transform_id": (
                None if scenario_record is None else scenario_record["transform_id"]
            ),
            "policy_seed": (
                None
                if reset_response.get("policy_seed") is None
                else int(reset_response["policy_seed"])
            ),
            "checkpoint": str(reset_response["checkpoint"]),
            "steps": step_index,
            "episode_seconds": round(episode_seconds, 6),
            "model_load_seconds": round(float(reset_response["load_seconds"]), 6),
            "remote_roundtrip_mean_seconds": round(
                statistics.fmean(roundtrip_seconds), 6
            ),
            "remote_roundtrip_warm_mean_seconds": round(
                statistics.fmean(roundtrip_seconds[2:]), 6
            ),
            "policy_preprocess_mean_seconds": round(
                statistics.fmean(preprocessing_seconds), 6
            ),
            "policy_inference_mean_seconds": round(
                statistics.fmean(inference_seconds), 6
            ),
            "policy_inference_first_two_seconds": [
                round(value, 6) for value in inference_seconds[:2]
            ],
            "policy_inference_warm_mean_seconds": round(
                statistics.fmean(inference_seconds[2:]), 6
            ),
            "policy_inference_warm_median_seconds": round(
                statistics.median(inference_seconds[2:]), 6
            ),
            "policy_postprocess_mean_seconds": round(
                statistics.fmean(postprocessing_seconds), 6
            ),
            "transport_and_serialization_warm_mean_seconds": round(
                statistics.fmean(
                    roundtrip - preprocessing - inference - postprocessing
                    for roundtrip, preprocessing, inference, postprocessing in zip(
                        roundtrip_seconds[2:],
                        preprocessing_seconds[2:],
                        inference_seconds[2:],
                        postprocessing_seconds[2:],
                    )
                ),
                6,
            ),
            "environment_step_mean_seconds": round(
                statistics.fmean(environment_seconds), 6
            ),
            "reward_sum": round(sum(rewards), 6),
            "final_success": final_success,
            "success_ever": success_ever,
            "environment_action_sha256": action_hasher.hexdigest(),
            "first_environment_action": first_environment_action,
            "initial_observation_sha256": initial_observation_sha256,
            "observation_sequence_sha256": observation_hasher.hexdigest(),
            "terminated": terminated_value,
            "terminated_ever": terminated_ever,
            "policy_terminated_ever": policy_terminated_ever,
            "truncated": truncated_value,
            "status": "pass",
        }
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
