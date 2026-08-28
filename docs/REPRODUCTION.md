# Reproducing the numerical results

Extract the source ZIP and run commands from the directory containing
`README.md`. Numerical reproduction uses only CPython 3.11 and its standard
library. It does not download policies or run a simulator.

## Fast reconstruction of all reported studies

```bash
python3 scripts/verify_source.py
python3 scripts/reproduce_paper.py --mode stored
python3 scripts/reproduce_environment_seeds.py --mode stored
```

The first command runs 46 tests and small executable examples. The next two
recompute means and paired statistics from stored individual repetitions.
Outputs are `rerun/paper/result.json` and
`rerun/environment-seeds/result.json`. These are stored reconstructions,
not newly executed allocation paths.

| Study | Recorded input | Repetitions and inference | Primary budgets |
| --- | --- | --- | --- |
| Mechanism | 272 scenarios × 3 policies | 200 paired scenario permutations | 48, 96, 192 |
| Preference | Same recorded tensor | 200 permutations; original EIG/.25 anchors | 48, 96, 192 |
| Ranking comparators | Independent 272-scenario tensor | 200 paired permutations | 48, 96, 192 |
| Environment-seed stability | 136 scenarios × 3 policies × 12 seeds | 50 paths per seed; inference over 12 seed means | 24, 48, 96 |

Budgets include the initial 12 observations. Main-study results are not
12-seed averages; the seed study does not treat 600 nested paths as 600
independent environment samples.

The seed archive includes original source, three active raw acquisitions,
tensor, 12 replay files, and merged report. Its verifier reconstructs the
report independently. The public command additionally rebuilds the tensor
from raw outcomes and checks every full-endpoint metric. Only raw file hashes
differ because local path metadata was anonymized; both original and released
hashes are checked separately, without relabelling the rebuilt input hashes.
All five programs remain visible. Its original mixed scientific result
(`PARTIAL`) is preserved, distinct from successful data verification.

Preference `.50/.75` row files are dated **trace-verified reconstructions**,
not recovered original shards. They were recomputed and matched against all
400 original action hashes, budget means, and paired statistics, including
task-switch counts. Original summary/trace evidence remains a separate
oracle. The `.00/.25` anchors are original mechanism-study rows, as in the
reported analysis.

## Re-run the allocation algorithms

Try two repetitions first (a smoke run, not the full study):

```bash
python3 scripts/reproduce_paper.py --mode replay --study ranking --repetitions 2 --workers 2 --output rerun/ranking-smoke
```

Then run all primary-budget repetitions:

```bash
python3 scripts/reproduce_paper.py --mode replay --study mechanism --workers 4 --output rerun/mechanism
python3 scripts/reproduce_paper.py --mode replay --study preference --workers 4 --output rerun/preference
python3 scripts/reproduce_paper.py --mode replay --study ranking --workers 4 --output rerun/ranking
python3 scripts/reproduce_environment_seeds.py --mode replay --workers 4 --output rerun/seeds
```

`--study all` runs the first three paths, but a single platform need not
match every historical floating-point tie. Use these validated runtimes for
exact historical comparisons; all platforms use the same source:

| Fresh validation, 2026-08-28 | Runtime | Matched coverage |
| --- | --- | --- |
| Mechanism | Windows x86-64, CPython 3.11.15 | 200 × 3 programs × 3 budgets × 5 metrics = 9,000 values |
| Preference `.50/.75` | macOS arm64, CPython 3.11.3 | 400 original traces, 328 summary/statistic fields; 6,000 independent count checks |
| Ranking comparators | macOS arm64, CPython 3.11.3 | 200 × 5 programs × 3 budgets × 5 metrics = 15,000 values |
| Environment seeds | macOS arm64, CPython 3.11.3 | 12 × 50 paths × 5 programs × 3 budgets × 5 metrics = 45,000 values |

All comparisons above had zero metric error. Mechanism/preference/ranking
metrics are also reconstructed independently from actual revealed cells
using rational arithmetic. Full original action digests are compared only
when execution reaches their original digest budget; prefix runs do not
claim an unexecuted full trajectory.

Near-tied information scores can differ across platform math libraries.
Windows produced two EIG Gap-MAE row differences in the ranking study
(repetitions 96 and 172 at B96), while every other value matched; Mac matched
all values. Conversely, the mechanism EIG traces match the validated Windows
runtime. No epsilon, rounding, alternate tie rule, or replacement reference
value is used to hide differences. Mismatch returns a nonzero exit code and
preserves actual output. Stored reconstruction is available without changing
runtime.

Commands write `progress.json`, individual `cases/`, and a final `result.json`.
Repeat the same command with `--resume` after interruption. Changed source,
input, runtime, or cutoff requires a new output directory. `--workers`
controls ordinary CPU processes. The primary-budget main-study runs took
roughly a minute each with 8–12 workers in validation, not a timing guarantee.

## Optional longer trajectories

```bash
python3 scripts/reproduce_paper.py --mode replay --study mechanism --through 816 --workers 4 --output rerun/mechanism-complete
python3 scripts/reproduce_paper.py --mode replay --study ranking --through 816 --workers 4 --output rerun/ranking-complete
python3 scripts/reproduce_environment_seeds.py --mode replay --through 408 --workers 4 --output rerun/seeds-complete
```

`--through 408` is also available for the first two commands; the seed command
offers `--through 204`. Preference response remains a B192 study. Complete
trajectories can be much slower: historical selection times total tens of
CPU-hours across the main studies. They were not all freshly rerun for this
source update. Their original rows and digests are included as stored evidence.

Prefix capture stops at an original checkpoint without changing the
allocator's horizon or comparison rule. SRank's horizon remains 204 in the
ranking comparison and 102 in the seed study. The seed study keeps its
original local-capacity Saad adapter; the later independent comparison uses
global U=1536. They must not be silently interchanged.

## Generate and analyze new simulator outcomes

Follow [the simulator guide](../simulation/README.md) for original policy
servers, episode runner, scenarios, and external software/models. No fresh
simulator installation or GPU rerun is claimed for this source update.

Before acquisition, copy the relevant `simulation/*-acquisition-spec.json`
into your run directory and set the exact RT-1-X checkpoint path used by the
server. Do not edit raw output to make a checkpoint match. Keep scenario
membership and policy seeds unchanged. After all three batches finish:

```bash
python3 simulation/build_tensor.py --study mechanism --acquisition-spec rerun/acquisition.json --small rerun/raw/small.json --base rerun/raw/base.json --rt1x rerun/raw/rt1x.json --output rerun/new-tensor.json
python3 scripts/reproduce_paper.py --mode replay --study mechanism --custom-tensor rerun/new-tensor.json --workers 4 --output rerun/new-results
```

For the other scenario set, use `ranking` in both commands and its acquisition
specification. Original validators reject incomplete/duplicate rows, wrong
scenarios/checkpoints/seeds, and non-Boolean success. New truth rates are
computed from the new observations. This route never reads paper expectations
and reports `reference_comparison: not_applicable`; new outcomes need not
equal the historical tensor. Conversion of all six original raw batches
reproduced every supplied scenario record exactly in the release check.

## Implementation and provenance

- `scripts/fixed_preference.py`, `scripts/ranksplit.py`: reusable scores.
- `vendor/ranksplit/scripts/`: byte-identical original numerical/study code.
- `scripts/paper_reproduction.py`: checkpoint adapters, independent
  observation oracles, and post-computation comparisons.
- `data/paper/manifest.json`: input/source/reference hashes. Expected values
  reside separately and are not allocator inputs.
- `data/environment-seeds-provenance.json`: original release identity.
  Local host/path metadata was anonymized with original/released hashes
  recorded. All scientific source, tensor, replay and merged-result bytes
  remain unchanged, as do every raw observation and its scientific metadata.

Tests include analytic metric oracles, full-run/prefix agreement, forbidden
expected-result access during allocation, altered outcomes, invalid raw
batches, reordered/missing repetitions, and changed resume inputs. Expected
constants in tests are assertions, not score inputs.
