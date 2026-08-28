# Source release notes

## 2026-08-28: expanded paper reproduction

Added stored reconstruction and fresh recorded-outcome replay for the
mechanism, preference, ranking-comparator, and 12-environment-seed studies.
Added raw-batch conversion and a custom-outcome route. The PDF and original
scientific kernels/results were not changed.

All primary-paper-budget repetitions were re-executed: 9,000 mechanism
metric values on Windows CPython 3.11.15, 15,000 ranking values on macOS
CPython 3.11.3, and 45,000 seed-study values on that Mac runtime matched
exactly. Preference matched 400 original action digests and 328 aggregate,
paired, and telemetry fields exactly, plus 6,000 independent count checks.
Full original trajectories remain stored evidence; fresh prefix checks do
not claim they were re-executed.

Preference rows are dated trace-verified reconstructions with original
summary/trace evidence retained separately. The seed archive keeps all 39
payloads numerically unchanged; local host/path metadata was anonymized with
both original and released hashes recorded. Its
mixed scientific result (`PARTIAL`) and all comparators remain visible.
Both historical raw-to-tensor conversions matched all scenario records.
The expanded verifier runs 46 tests. No new GPU simulation was performed.

See [REPRODUCTION.md](REPRODUCTION.md) for commands and runtime details.
Two EIG ranking-study Gap-MAE rows differ on Windows but match on Mac;
neither formula, tie rule, expected value, nor tolerance was changed to
force agreement.

## 2026-08-28: reviewer source snapshot

This release supplies the RankSplit calculation modules, the recorded binary
outcomes for the first comparison, its replay commands, and optional original
simulation integration source with both scenario sets. The anonymous mirror
can retain the same URL as documentation and reproduction coverage improve.
Keep the downloaded ZIP when reviewing a particular version.

The numerical kernels, allocation rules, original outcome tensor, and reported
reference values are unchanged in this packaging update. Additional reference
rows are comparison oracles, read only after the runtime probe computes its
results. They are not inputs to the allocation rule.

Verification performed for this snapshot:

- 32 focused source tests, a deterministic quickstart, and one reference replay
  pass on Mac CPython 3.11.3.
- The independent EIG runtime probe matches all 200 repetitions, five budgets,
  five metrics, and complete allocation hashes in the historical Windows
  CPython 3.11.15 environment.
- Optional simulator source hashes, Python syntax, both scenario manifests,
  their physical-pose disjointness, and corrupt-input rejection are checked.
  No fresh GPU simulation or clean simulator installation is claimed.

The existing 200-repetition aggregate verifier is a smoke check with a 2%
tolerance. It is not an exact-path certification; near-tied EIG scores change
some Mac paths relative to the historical Windows runtime. No numerical
formula or stored value has been changed to force agreement.

At this initial snapshot, not yet included in the executable reviewer path were the preference
sensitivity, the second ranking-comparator study, the environment-seed study,
and automatic conversion of newly simulated raw outputs into a replay tensor.
Those paths were subsequently added in the expanded entry above. The initial
snapshot and original experimental evidence remain unchanged.
