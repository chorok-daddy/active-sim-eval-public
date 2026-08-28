"""Serve official RT-1-X WidowX actions from WSL over the audit TCP bridge."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import struct
import time

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import tensorflow as tf


CHECKPOINT = os.environ.get(
    "ACTIVE_SIM_EVAL_RT1_CHECKPOINT",
    "/home/researcher/checkpoints/rt_1_x_tf_trained_for_002272480_step",
)
SIMPLER_ROOT = Path(
    os.environ.get(
        "ACTIVE_SIM_EVAL_SIMPLER_ROOT",
        "/mnt/c/Users/User/src/SimplerEnv-ms3",
    )
)
HOST = "0.0.0.0"
PORT = 8765


def recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed while receiving payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_request(connection: socket.socket) -> tuple[dict[str, object], bytes]:
    metadata_size, payload_size = struct.unpack("!IQ", recv_exact(connection, 12))
    metadata = json.loads(recv_exact(connection, metadata_size))
    payload = recv_exact(connection, payload_size) if payload_size else b""
    return metadata, payload


def send_response(connection: socket.socket, response: dict[str, object]) -> None:
    payload = json.dumps(response).encode("utf-8")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def load_rt1_class() -> type:
    module_path = (
        SIMPLER_ROOT / "simpler_env" / "policies" / "rt1" / "rt1_model.py"
    )
    spec = importlib.util.spec_from_file_location("active_sim_eval_rt1_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RT-1 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RT1Inference


class RT1BridgePolicy:
    def __init__(self) -> None:
        tf.keras.utils.set_random_seed(0)
        tf.config.experimental.enable_op_determinism()
        rt1_inference = load_rt1_class()
        started = time.perf_counter()
        self.model = rt1_inference(
            saved_model_path=CHECKPOINT,
            policy_setup="widowx_bridge",
            action_scale=1.0,
        )
        self.load_seconds = time.perf_counter() - started

    def reset(self, instruction: str) -> float:
        started = time.perf_counter()
        self.model.reset(instruction)
        return time.perf_counter() - started

    def step(self, image: np.ndarray) -> tuple[np.ndarray, bool, float]:
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"expected one HWC RGB image, got {image.shape}")
        started = time.perf_counter()
        _raw_action, action = self.model.step(image)
        elapsed = time.perf_counter() - started
        environment_action = np.concatenate(
            (
                np.asarray(action["world_vector"]),
                np.asarray(action["rot_axangle"]),
                np.asarray(action["gripper"]),
            ),
            axis=-1,
        ).reshape(1, 7)
        terminate_values = np.asarray(action["terminate_episode"]).reshape(-1)
        predicted_terminate = bool(terminate_values[0] > 0)
        return environment_action, predicted_terminate, elapsed


def main() -> None:
    policy = RT1BridgePolicy()
    print(
        json.dumps(
            {
                "status": "ready",
                "host": HOST,
                "port": PORT,
                "checkpoint": CHECKPOINT,
                "load_seconds": round(policy.load_seconds, 6),
                "devices": [
                    str(device) for device in tf.config.list_physical_devices()
                ],
                "deterministic_ops": True,
            }
        ),
        flush=True,
    )

    stop = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)
        while not stop:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        metadata, payload = receive_request(connection)
                    except ConnectionError:
                        break
                    request_type = metadata["type"]
                    if request_type == "reset":
                        reset_seconds = policy.reset(str(metadata["instruction"]))
                        send_response(
                            connection,
                            {
                                "status": "reset",
                                "checkpoint": CHECKPOINT,
                                "load_seconds": policy.load_seconds,
                                "reset_seconds": reset_seconds,
                                "policy_seed": None,
                            },
                        )
                    elif request_type == "step":
                        image = np.frombuffer(payload, dtype=np.uint8).reshape(
                            tuple(metadata["shape"])
                        )
                        action, predicted_terminate, inference_seconds = policy.step(
                            image
                        )
                        send_response(
                            connection,
                            {
                                "status": "ok",
                                "action": action.tolist(),
                                "action_format": "axis_angle_binary_gripper",
                                "predicted_terminate": predicted_terminate,
                                "preprocess_seconds": 0.0,
                                "inference_seconds": inference_seconds,
                                "postprocess_seconds": 0.0,
                                "timing_scope": "full RT1Inference.step",
                            },
                        )
                    elif request_type == "shutdown":
                        send_response(connection, {"status": "stopping"})
                        stop = True
                        break
                    else:
                        send_response(
                            connection,
                            {"status": "error", "message": "unknown request type"},
                        )


if __name__ == "__main__":
    main()
