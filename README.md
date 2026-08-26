# ActiveSimEval: RankSplit source

Dependency-light Python source for the RankSplit calculation surface. The
repository is organized for a reviewer who wants to go from a clean checkout
to a verified CPU smoke run in a few minutes.

> The public source release contains the reusable calculation modules, a
> compact binary-outcome tensor, tests, and reviewer-facing verification
> commands. It does not contain manuscripts, policy checkpoints, simulator
> environments, rendered figures, or private workspace paths.

## Quick start

```bash
git clone https://github.com/chorok-daddy/active-sim-eval-public.git
cd active-sim-eval-public
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

The full 200-repetition result check at the three primary budgets is available
when a reviewer wants the reported aggregate values:

```bash
python3 scripts/reproduce_results.py --full
```

The comparison uses a 2% relative tolerance and a `1e-8` absolute floor for
values close to zero.

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
  verify_source.py              one-command verifier
data/
  confirmation_tensor.json      compact public binary outcomes
  reported_results.json         reported metrics and tolerances
examples/
  ranksplit_quickstart.py       deterministic CPU-only example
tests/                           numerical, scoring, import, and smoke tests
docs/ONBOARDING.md               reviewer-oriented setup path
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
