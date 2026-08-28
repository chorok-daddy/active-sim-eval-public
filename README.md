# ActiveSimEval: RankSplit source

Dependency-light Python source for the RankSplit calculation surface. The
repository is organized for a reviewer who wants to go from a clean checkout
to a verified CPU smoke run in a few minutes.

> The public source release contains the reusable calculation modules, a
> compact binary-outcome tensor, tests, and reviewer-facing verification
> commands. Optional original simulator integration source and scenario inputs
> are provided separately; model weights and simulator environments are not
> bundled.

To generate new robot outcomes instead of using the supplied data, see
[`simulation/README.md`](simulation/README.md). This optional GPU path is
separate from the default CPU reproduction and has not been rerun for this
source release.

## Quick start

Download the source ZIP and extract it. From the extracted repository root
(the directory containing this README), run:

```bash
python3 --version
python3 scripts/verify_source.py
```

Expected final line:

```text
PUBLIC SOURCE VERIFY PASS
```

The verifier runs the focused unit tests, the deterministic quickstart, and a
one-repetition numeric check against the reported primary-budget rows. No
package installation, simulator download, GPU, or external data is needed.

## What is being verified?

The one-command check covers:

1. exact Beta--Bernoulli probability and information calculations;
2. fixed-preference and RankSplit scoring invariants;
3. package-style and script-directory imports;
4. the deterministic CPU quickstart and its expected selection;
5. one deterministic replay of the public outcome tensor.

To recompute the first comparison over 200 paired repetitions at its three
primary budgets, run:

```bash
python3 scripts/reproduce_results.py --full
```

This existing smoke verifier uses a 2% relative tolerance and a `1e-8`
absolute floor. Its PASS is not a strict trace or full-paper certification.
The sensitivity study, second ranking-comparator comparison, and environment
seed study are not yet covered by this command; `--full` means all repetitions
of the first comparison, not all experiments in the paper.

An independent EIG runtime check compares all five metrics and the full
allocation trace for each of 200 repetitions:

```bash
python3 scripts/check_replay_runtime.py --all
```

This check matched the historical Windows CPython 3.11.15 environment exactly.
On Mac CPython 3.11.3, near-tied floating-point EIG scores can change individual
choices; the check reports these differences rather than adjusting the source
or reference values. This check covers EIG only. See
[`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md) for the release scope.

For the step-by-step reviewer path, see
[`docs/ONBOARDING.md`](docs/ONBOARDING.md).

## Requirements

- CPython 3.11.15, recorded in [`.python-version`](.python-version)
- Python standard library only
- A shell that can run the commands above

## Repository map

```text
scripts/
  fixed_preference.py           exact Beta--Bernoulli calculations
  ranksplit.py                  clarity-adaptive RankSplit score
  reproduce_results.py          tensor replay and result check
  check_replay_runtime.py        EIG per-repetition metrics and trace check
  verify_source.py              one-command verifier
data/
  confirmation_tensor.json      compact public binary outcomes
  reported_results.json         reported metrics and tolerances
examples/
  ranksplit_quickstart.py       deterministic CPU-only example
tests/                           numerical, scoring, import, and smoke tests
docs/ONBOARDING.md               reviewer-oriented setup path
simulation/                      optional original simulator source and inputs
```

The modules support both `scripts.ranksplit` imports from the repository root
and the legacy `scripts/` directory import used by the example and tests.

## Run individual checks

```bash
python3 -m unittest -q tests.test_fixed_preference
python3 -m unittest -q tests.test_ranksplit
python3 -m unittest -q tests.test_ranksplit_quickstart
python3 -m unittest -q tests.test_ranksplit_source_imports
python3 examples/ranksplit_quickstart.py
python3 scripts/reproduce_results.py
```

## Troubleshooting

- Run commands from the repository root.
- Confirm that `python3 --version` reports Python 3.11.x.
- If an import fails, run `python3 scripts/verify_source.py` before installing
  anything; the supported path has no third-party runtime dependency.

## Citation

For this source release, cite the repository and the verified commit:

```bibtex
@software{lee2026activesimeval_ranksplit,
  author  = {Ki-Baek Lee},
  title   = {ActiveSimEval: RankSplit source},
  year    = {2026},
  url     = {https://github.com/chorok-daddy/active-sim-eval-public}
}
```

## License

The source in this repository is provided under the MIT License. See
[`LICENSE`](LICENSE). Any separately supplied third-party software, data, or
models remain governed by their own terms; none are included in this source
release.
