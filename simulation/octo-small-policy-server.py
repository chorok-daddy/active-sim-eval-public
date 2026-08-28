"""Serve SIMPLER-compatible Octo-small 1.0 inference from WSL over TCP.

The server keeps image and action-ensemble history across an episode. It sends
unnormalized Bridge-policy actions to a native-Windows simulator client. The
transport is an infrastructure bridge, not a new policy method.
"""

from __future__ import annotations

from collections import deque
import json
import os
import socket
import struct
import time

# Closed-loop robot rollouts amplify tiny GPU differences. Require XLA's
# deterministic GPU kernels unless the caller already supplied this flag.
if "xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + " --xla_gpu_deterministic_ops=true"
    ).strip()

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


CHECKPOINT = os.environ.get(
    "ACTIVE_SIM_EVAL_OCTO_CHECKPOINT", "hf://rail-berkeley/octo-small"
)
DATASET_ID = "bridge_dataset"
HOST = "0.0.0.0"
PORT = 8765
PREDICTION_HORIZON = 4


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


class OctoBridgePolicy:
    def __init__(self, model: object | None = None) -> None:
        # TensorFlow is used for checkpoint/text utilities; JAX owns the GPU.
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # A shared-model equivalence fixture may initialize TensorFlow
            # before constructing this wrapper. It is safe only if the GPU is
            # already hidden exactly as required.
            if tf.config.get_visible_devices("GPU"):
                raise
        from octo.model.octo_model import OctoModel

        if model is None:
            started = time.perf_counter()
            self.model = OctoModel.load_pretrained(CHECKPOINT)
            self.load_seconds = time.perf_counter() - started
        else:
            self.model = model
            self.load_seconds = 0.0
        self.action_mean = jnp.asarray(
            self.model.dataset_statistics[DATASET_ID]["action"]["mean"]
        )
        self.action_std = jnp.asarray(
            self.model.dataset_statistics[DATASET_ID]["action"]["std"]
        )
        self.rng = self._initialize_rng(0)
        self.policy_seed = 0
        self.image_history: deque[jax.Array] = deque(maxlen=2)
        self.action_history: deque[jax.Array] = deque(
            maxlen=PREDICTION_HORIZON
        )
        self.task = None

    @staticmethod
    def _initialize_rng(seed: int) -> jax.Array:
        rng = jax.random.PRNGKey(seed)
        for _ in range(5):
            rng, _ = jax.random.split(rng)
        return rng

    def reset(self, instruction: str, policy_seed: int | None = None) -> None:
        if policy_seed is not None:
            self.policy_seed = policy_seed
            self.rng = self._initialize_rng(policy_seed)
        self.task = self.model.create_tasks(texts=[instruction])
        self.image_history.clear()
        self.action_history.clear()

    def step(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        preprocess_started = time.perf_counter()
        image_jax = jnp.asarray(image)
        image_jax = jax.image.resize(
            image_jax,
            shape=(image.shape[0], 256, 256, 3),
            method="lanczos3",
            antialias=True,
        )
        image_jax = jnp.clip(jnp.round(image_jax), 0, 255).astype(jnp.uint8)
        self.image_history.append(image_jax)
        images = jnp.stack(tuple(self.image_history), axis=1)
        pad_mask = jnp.ones(images.shape[:2], dtype=jnp.float32)
        # JAX dispatch is asynchronous. Synchronize every timed stage so the
        # reported component costs are attributable rather than shifted into
        # the following stage.
        images.block_until_ready()
        pad_mask.block_until_ready()
        preprocess_seconds = time.perf_counter() - preprocess_started

        self.rng, key = jax.random.split(self.rng)
        inference_started = time.perf_counter()
        normalized = self.model.sample_actions(
            {"image_primary": images, "pad_mask": pad_mask},
            self.task,
            rng=key,
        )
        raw_predictions = normalized * self.action_std[None] + self.action_mean[None]
        raw_predictions.block_until_ready()
        inference_seconds = time.perf_counter() - inference_started

        postprocess_started = time.perf_counter()
        self.action_history.append(raw_predictions)
        count = len(self.action_history)
        current_predictions = jnp.stack(
            [
                prediction[:, index]
                for index, prediction in zip(
                    range(count - 1, -1, -1), self.action_history
                )
            ]
        )
        ensembled = jnp.mean(current_predictions, axis=0)
        ensembled.block_until_ready()
        action = np.asarray(ensembled)
        postprocess_seconds = time.perf_counter() - postprocess_started
        return (
            action,
            preprocess_seconds,
            inference_seconds,
            postprocess_seconds,
        )


def main() -> None:
    policy = OctoBridgePolicy()
    print(
        json.dumps(
            {
                "status": "ready",
                "host": HOST,
                "port": PORT,
                "checkpoint": CHECKPOINT,
                "load_seconds": round(policy.load_seconds, 6),
                "devices": [str(device) for device in jax.devices()],
                "xla_flags": os.environ.get("XLA_FLAGS", ""),
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
                        requested_seed = metadata.get("policy_seed")
                        policy.reset(
                            str(metadata["instruction"]),
                            None if requested_seed is None else int(requested_seed),
                        )
                        send_response(
                            connection,
                            {
                                "status": "reset",
                                "checkpoint": CHECKPOINT,
                                "load_seconds": policy.load_seconds,
                                "policy_seed": policy.policy_seed,
                            },
                        )
                    elif request_type == "step":
                        image = np.frombuffer(payload, dtype=np.uint8).reshape(
                            tuple(metadata["shape"])
                        )
                        (
                            action,
                            preprocess_seconds,
                            inference_seconds,
                            postprocess_seconds,
                        ) = policy.step(image)
                        send_response(
                            connection,
                            {
                                "status": "ok",
                                "action": action.tolist(),
                                "action_format": "euler_gripper_probability",
                                "predicted_terminate": False,
                                "preprocess_seconds": preprocess_seconds,
                                "inference_seconds": inference_seconds,
                                "postprocess_seconds": postprocess_seconds,
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
