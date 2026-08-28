# Optional: regenerate the 12-seed simulator study

This is the slower, advanced route for generating new observations, not a
requirement for CPU result verification. It uses 136 official scenarios,
three policy checkpoints, and environment seeds 100 through 111:
4,896 episodes. Together with the two main scenario sets, full acquisition
requires 6,528 episodes. Analysis repetitions reuse these observations.

Follow [the simulator and model setup](README.md) first. The original source
and scenarios for this study are bundled inside `data/environment-seeds.zip`.
Extract and verify them with the ordinary CPU command, from the repository
root:

```powershell
python scripts/reproduce_environment_seeds.py --mode stored --output rerun/seeds-reference
$seedSource = "rerun/seeds-reference/inputs/source"
$seedProtocol = "$seedSource/configs/protocols/2026-08-06-ranksplit-environment-seed-transfer.pre-result.json"
```

The `seeds-reference` directory contains supplied evidence; keep new results
under `rerun/new-seeds/` instead. Do not alter the extracted source or its
protocol. The commands below use the original simulator, tensor builder,
replay, and summary programs, not the wrapper that compares historical data.

## Acquire new observations

Start one policy server at a time, using the scripts in the extracted
`source/scripts/` directory and the settings in the main simulator guide.
For this original seed-study validator, the RT-1-X checkpoint identifier is
the literal WSL path
`/home/researcher/checkpoints/rt_1_x_tf_trained_for_002272480_step`.
Install or mount/symlink the actual upstream checkpoint there and pass that
same path as `ACTIVE_SIM_EVAL_RT1_CHECKPOINT` before starting the server.
This is an original path-metadata constraint, not a restriction on outcomes;
do not rewrite a raw file to disguise a different checkpoint path.

From the native-Windows simulator terminal, after the server reports ready:

```powershell
python "$seedSource/scripts/run-policy-family-environment-batch.py" --policy-name octo-small --environment-seeds 100,101,102,103,104,105,106,107,108,109,110,111 --manifest "$seedSource/configs/manifests/2026-07-14-task-profile-follow-up-scenarios.json" --policy-seed 0 --simpler-root C:/path/to/SimplerEnv --output rerun/new-seeds/raw/octo-small.json --checkpoint-every 1 --resume --shutdown-server
```

Repeat for `octo-base` and `rt-1-x`, changing the server checkpoint, policy
name, and output filename each time. For RT-1-X, omit `--policy-seed`: this
runner retains a zero wire-request seed but records the server's actual
`null` policy seed. Each policy produces 1,632 observations. The runner saves
after each episode, and the same command resumes the same configuration.
Try a single episode before a full batch, as in the main guide; for this
official-scenario study use `--seed 100 --episode-id 0` and omit both
`--scenario-manifest` and `--scenario-id` in the episode-client command.

## Analyze the new observations

Once all three batches are complete, use the CPU environment to build a new
tensor. Its outcomes and success rates come from the new raw files:

```powershell
python "$seedSource/scripts/build-ranksplit-environment-seed-transfer-tensor.py" --protocol $seedProtocol --small rerun/new-seeds/raw/octo-small.json --base rerun/new-seeds/raw/octo-base.json --rt1 rerun/new-seeds/raw/rt-1-x.json --output rerun/new-seeds/tensor.json
```

Replay all 50 paired paths for each seed, then summarize across the 12 seeds:

```powershell
foreach ($evaluationSeed in 100..111) {
  python "$seedSource/scripts/replay-ranksplit-environment-seed-transfer.py" --protocol $seedProtocol --tensor rerun/new-seeds/tensor.json --environment-seed $evaluationSeed --output "rerun/new-seeds/replay-$evaluationSeed.json"
  if ($LASTEXITCODE -ne 0) { throw "Replay failed for seed $evaluationSeed" }
}
$seedReplays = 100..111 | ForEach-Object { "rerun/new-seeds/replay-$_.json" }
python "$seedSource/scripts/merge-ranksplit-environment-seed-transfer-replays.py" --protocol $seedProtocol --tensor rerun/new-seeds/tensor.json --shards $seedReplays --output rerun/new-seeds/summary.json
```

These original replay commands run the complete B408 trajectories, including
all five comparison programs. They are slower than the primary-budget prefix
commands in the main reproduction guide. A completed per-seed file can be
kept after interruption; an unfinished seed must be rerun. Never merge an
old-data replay into the new-data run: the summary program checks input hashes.

The new summary evaluates the new observations; it does not force agreement
with the paper's stored outcomes or conclusions. This release checked source
integrity, original raw-to-tensor reconstruction, and CPU replay, but did not
perform a new GPU acquisition or a clean installation of these GPU environments.
