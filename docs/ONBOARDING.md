# Reviewer onboarding

This page is the shortest path from a fresh checkout to a verified source
run. It is intentionally separate from the paper so that a reviewer can use
the code without reading project history first.

## 1. Prepare Python

Use CPython 3.11.x. The preferred patch version is recorded in the repository
root:

```bash
cat .python-version
python3 --version
```

The source path uses only the standard library. Do not create an environment
or install packages for the default check.

## 2. Run the one-command check

From the repository root:

```bash
python3 scripts/verify_source.py
```

The command runs:

- focused unit tests;
- package-style imports through `scripts.*`;
- direct imports through `scripts/`;
- the deterministic quickstart example;
- one deterministic result-tensor replay.

The final line should be:

```text
PUBLIC SOURCE VERIFY PASS
```

The example also prints a JSON object. The expected selection is
`drawer/policy-2`. The result-tensor check is run separately so the example
remains easy to inspect.

## 3. Inspect the public API

From the repository root:

```bash
python3 - <<'PY'
from scripts.fixed_preference import BetaPosterior, ranking_pd_eig
from scripts.ranksplit import ranksplit_score

states = (
    BetaPosterior(3, 4),
    BetaPosterior(5, 3),
    BetaPosterior(2, 6),
)
print(ranksplit_score(states, observed_policy=1, preference_lambda=0.5))
print(ranking_pd_eig(states, observed_policy=1, preference_lambda=0.5))
PY
```

The output is a finite numeric score and an
`InformationDecomposition` value. Exact values can vary only if the source
or Python numerical behavior changes; the repository tests provide the
supported invariants.

## 4. Understand the source layout

| Path | Purpose |
| --- | --- |
| `scripts/fixed_preference.py` | Beta--Bernoulli information and fixed-preference ranking calculations |
| `scripts/ranksplit.py` | Clarity-adaptive RankSplit score |
| `scripts/reproduce_results.py` | Public tensor replay and metric check |
| `data/confirmation_tensor.json` | Compact binary outcomes used by the replay |
| `data/reported_results.json` | Reported metrics and comparison tolerances |
| `examples/ranksplit_quickstart.py` | Minimal deterministic example |
| `tests/` | Numerical, scoring, import, and quickstart checks |
| `scripts/verify_source.py` | Combined verification entry point |

## 5. Reproduction boundary

The public replay starts from already recorded binary outcomes; it does not
download models or recreate simulator episodes. The default check verifies a
deterministic reference repetition. Use the following command to recompute
the reported 200-repetition aggregate metrics at the three primary budgets:

```bash
python3 scripts/reproduce_results.py --full
```

The aggregate comparison accepts 2% relative deviation and `1e-8` absolute
deviation near zero to accommodate ordinary floating-point or stochastic
variation.

## 6. If something fails

1. Confirm that the command is being run from the repository root.
2. Confirm Python 3.11.x with `python3 --version`.
3. Run one test module to isolate the failure:

   ```bash
   python3 -m unittest -q tests.test_ranksplit
   ```

4. Record the Python version, operating system, command, and complete error
   output when reporting an issue.

## 7. License and source use

The repository source is distributed under the MIT License in the root
[`LICENSE`](../LICENSE) file. No third-party package or data is bundled in the
default source path.
