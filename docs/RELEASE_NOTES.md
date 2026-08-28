# Source release notes

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

Not yet included in the executable reviewer path are the preference
sensitivity, the second ranking-comparator study, the environment-seed study,
and automatic conversion of newly simulated raw outputs into a replay tensor.
The optional simulation guide generates raw outputs and describes this
boundary. Future additions should be dated here and should not overwrite
the original experimental evidence.
