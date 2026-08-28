# Optional: generate new simulator outcomes

The default CPU reproduction starts from the recorded outcomes supplied in
`data/`. This directory additionally provides the original episode runner,
policy servers, and scenario inputs so that a reviewer can generate new
outcomes. It is not necessary to run a simulator to check the paper's numerical
analysis.

The five Python files and two scenario files here are byte-for-byte copies of
the experiment sources, listed in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).
The package checks their syntax and scenario consistency. A fresh GPU
simulation was **not** performed for this source release. New trajectories can
depend on the hardware and numerical environment; compare new outcomes as a
new run, not as a required byte-identical copy of the supplied data.

## Software and models

The recorded setup separates a native-Windows ManiSkill3 simulator from policy
inference in WSL2. The two processes communicate over TCP port 8765. Use one
policy server at a time. The original servers bind to `0.0.0.0`; run them only
on a trusted local machine/network and do not expose this port to the internet.

1. Install the [ManiSkill3 version of SIMPLER](https://github.com/simpler-env/SimplerEnv/tree/maniskill3),
   including the WidowX assets required by its four manipulation tasks. The
   SIMPLER checkout inspected on the experiment machine is commit
   `52a5088ca4bfc3a7159828af08874a198d6f95c3` and has no local changes.
   Do not substitute the older ManiSkill2 environments on the `main` branch.
2. Use separate policy environments for Octo and RT-1-X. Follow the upstream
   [policy installation instructions](https://github.com/simpler-env/SimplerEnv/tree/maniskill3#full-installation-rt-1-and-octo-inference).
   The inspected Octo checkout is unchanged at revision
   `653c54acde686fde619855f2eac0dd6edad7116b` (Octo 1.0).
   The inspected version reference is in
   [environment-reference.json](environment-reference.json). It is a read-only
   inventory of the existing environments on 2026-08-28, not a newly tested
   installation lock or proof that every package was unchanged since acquisition.
3. Obtain the [Octo-small](https://huggingface.co/rail-berkeley/octo-small) and
   [Octo-base](https://huggingface.co/rail-berkeley/octo-base) checkpoints through
   the upstream Octo loader. Use the RT-1-X TensorFlow checkpoint
   `rt_1_x_tf_trained_for_002272480_step` linked in the upstream SIMPLER
   [RT-1 instructions](https://github.com/simpler-env/SimplerEnv/tree/maniskill3#rt-1-inference-setup).
   No model weights or third-party environment assets are redistributed here;
   their original licenses apply.

The recorded machine has an NVIDIA RTX 4090 with 24 GB VRAM. This is a tested
hardware reference, not a measured minimum requirement. The simulator uses
`physx_cpu` and `sapien_cpu`; policy inference uses its separate GPU environment.
The default CPU result replay needs none of these packages or model downloads.

## Start a policy server

Run from this repository in the appropriate WSL policy environment. For
Octo-small:

```bash
ACTIVE_SIM_EVAL_OCTO_CHECKPOINT=hf://rail-berkeley/octo-small \
  python simulation/octo-small-policy-server.py
```

For Octo-base, use the same script with
`ACTIVE_SIM_EVAL_OCTO_CHECKPOINT=hf://rail-berkeley/octo-base`.
For RT-1-X, set the two paths explicitly:

```bash
ACTIVE_SIM_EVAL_SIMPLER_ROOT=/path/to/SimplerEnv \
ACTIVE_SIM_EVAL_RT1_CHECKPOINT=/path/to/rt_1_x_tf_trained_for_002272480_step \
  python simulation/rt1-x-policy-server.py
```

Wait for the server's JSON `status: ready` message. The scripts retain their
original historical defaults, so explicitly supply the paths shown above.
Do not run more than one of these servers on the same port.

## Try one episode

In a second terminal with the simulator environment active, set `PYTHONPATH`
to your SIMPLER checkout. For example, in PowerShell:

```powershell
$env:PYTHONPATH = "C:/path/to/SimplerEnv"
python simulation/run-octo-small-policy-episode.py --env-id PutCarrotOnPlateInScene-v1 --seed 0 --episode-id 0 --policy-seed 0 --scenario-manifest simulation/scenarios/mechanism-comparison.json --scenario-id ranksplit-v2-confirmation-v1:PutCarrotOnPlateInScene-v1:support-inset-3over32:p0:q0
```

The client defaults to `127.0.0.1:8765`. Check WSL localhost connectivity if
it cannot reach the ready server. Its JSON result contains `final_success`,
the actual checkpoint identity, trajectory hashes, and timing. Success is the
final `info["success"]` value, not an early success or the `terminated` flag.
This one-episode check also exposes missing environment assets before a batch.

## Run a complete scenario set

From the simulator terminal:

```powershell
python simulation/run-policy-family-interstitial-batch.py --policy-name octo-small --manifest simulation/scenarios/mechanism-comparison.json --policy-seed 0 --simpler-root C:/path/to/SimplerEnv --output rerun/raw/mechanism-octo-small.json --checkpoint-every 1 --resume --shutdown-server
```

The runner writes progress after each completed scenario. Repeating the same
command with the same configuration resumes that output; it does not turn
failed or missing episodes into zeros. `--shutdown-server` stops the policy
server after the last episode. Start the next policy's server before running
its batch.

Run the three policies for each of these two sets:

| Scenario file | Purpose | Episodes per policy |
| --- | --- | ---: |
| `scenarios/mechanism-comparison.json` | EIG, decomposition, adaptation, and preference analysis | 272 |
| `scenarios/ranking-comparison.json` | Comparison with the ranking-method adaptations | 272 |

For Octo-base, change `--policy-name`, the output filename, and the server's
checkpoint. For RT-1-X, use `--policy-name rt-1-x` and **omit** `--policy-seed`;
RT-1-X does not expose per-episode policy-seed control. Never reuse an output
filename across different policies or scenario sets. A complete two-set run
contains 1,632 episodes, before any optional environment-seed analysis.

Keep newly generated files under `rerun/`, separate from the supplied paper
data. The original full-batch outputs include Boolean outcomes and scenario
identities; the numerical reproduction path must use those observed outcomes,
not replace them with the supplied results when they differ.

## Source-only checks

To convert complete batches into a new tensor, first copy
`mechanism-acquisition-spec.json` or `ranking-acquisition-spec.json` to your
run directory **before acquisition**, setting the exact RT-1-X checkpoint
path used by your server. Do not rewrite raw results to match an old path.

```bash
python3 simulation/build_tensor.py --study mechanism --acquisition-spec rerun/acquisition.json --small rerun/raw/small.json --base rerun/raw/base.json --rt1x rerun/raw/rt1x.json --output rerun/new-tensor.json
python3 scripts/reproduce_paper.py --mode replay --study mechanism --custom-tensor rerun/new-tensor.json --workers 4 --output rerun/new-results
```

Use `ranking` and its specification for the independent comparator scenario
set. Original raw-payload checks reject missing or inconsistent outcomes;
custom replay derives its target from the new observations, never from paper
expectations. See [the reproduction guide](../docs/REPRODUCTION.md).

These commands require only the default CPU Python environment and do not
start any simulator, policy process, or model download:

```bash
python3 -m unittest -v tests.test_simulation_source
python3 simulation/run-policy-family-interstitial-batch.py --help
```

They check file provenance, Python syntax, the two complete scenario sets, and
rejection of corrupted scenario descriptors. Passing them is not a claim of a
fresh rollout or a clean installation of the optional GPU environments.
