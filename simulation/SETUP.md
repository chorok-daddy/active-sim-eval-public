# Optional simulator setup helper

This is Stage 2 of the [two-stage guide](../README.md#two-stages-of-reproduction).
**Stage 1, paper-result reproduction on a CPU, needs none of this setup.**
This helper is for reviewers who also want to generate new robot outcomes.

It downloads pinned upstream source, creates separate environments, and fetches
the public models/assets used by the original integration. Commands are printed
without running them unless `--execute` is present. It does not install drivers,
change the OS, start policy servers, or launch experiments.

## Prerequisites and scope

The recorded arrangement uses native Windows simulation and WSL2 Linux/NVIDIA
policy inference. WSL's CUDA support does not provide ManiSkill rendering:
the [official support table](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support)
lists WSL rendering as unsupported. Native Linux/NVIDIA supports rendering
upstream, but this helper does not yet certify a native-Linux end-to-end setup.
Install Git, CPython 3.11, and
[Miniforge/Conda](https://github.com/conda-forge/miniforge) on the appropriate
sides first. WSL2 and an NVIDIA driver with working CUDA/Vulkan support must
already be configured; the helper does not alter them. The recorded hardware
reference is an RTX 4090 with 24 GB VRAM, not a measured minimum requirement.

The recipes use the [inspected package versions](environment-reference.json)
and upstream requirements. They are **not a complete historical lockfile or
a freshly GPU-certified clean installation**. Versions of every transitive
dependency are not pinned. The source/command tests do not certify rendering
or policy inference. Check one episode before starting a long run.

Keep Windows and WSL setup roots separate. Prefer extracting the source ZIP
inside the WSL Linux filesystem for policy environments. All new environments,
source checkouts, downloads, caches, and receipts stay under the selected
`--root`; existing unrecorded environments are not reused. Large model and
asset downloads need several GB beyond the small source ZIP. The RT-1-X
archive alone is 774,610,722 bytes before extraction; this is not the total
installation size. The read-only `check` command reports free disk space.

## A. Windows: simulator and scene assets

In PowerShell, from the extracted repository root, preview first:

```powershell
python simulation/setup_simulation.py check --root external/windows-setup
python simulation/setup_simulation.py sources --root external/windows-setup
python simulation/setup_simulation.py install --component simulator --root external/windows-setup
python simulation/setup_simulation.py assets --root external/windows-setup
```

Then run the selected steps:

```powershell
python simulation/setup_simulation.py sources --root external/windows-setup --execute
python simulation/setup_simulation.py install --component simulator --root external/windows-setup --execute
python simulation/setup_simulation.py assets --root external/windows-setup --execute
```

If Conda is not on PATH, add `--conda C:/path/to/miniforge3/Scripts/conda.exe`
to the install command; use the executable, not a `.bat`/`.cmd` wrapper.
The simulator's Pinocchio dependency is installed from conda-forge, matching
the inspected Windows environment; PyPI `pin==3.9.0` has no Windows wheel.

The asset step resolves the four task groups together, downloads their shared
scene assets once, and includes the separate WidowX250S robot assets.
It preserves the upstream URLs and scene checksum, and additionally records
the downloaded WidowX version-tag ZIP checksum. Verified ZIPs are extracted
to temporary directories and renamed to absent targets for Windows compatibility.
The helper requires a fresh `assets/` directory. If interrupted, move the incomplete
`external/windows-setup/assets` folder aside before repeating that step;
the helper never deletes it. Do not redirect it to a shared asset collection.

To use this setup in the original client commands:

```powershell
$setupRoot = (Resolve-Path external/windows-setup).Path
$env:MS_ASSET_DIR = "$setupRoot/assets"
$env:PYTHONPATH = "$setupRoot/src/SimplerEnv"
conda activate "$setupRoot/envs/simulator"
```

Use a Miniforge-enabled PowerShell for `conda activate`, or use
`conda run --no-capture-output --prefix "$setupRoot/envs/simulator" python ...`
instead of `python ...`. The first approach sets the required Windows DLL
search paths as well as selecting the interpreter.

## B. WSL2/Linux: policy environments and public models

From the extracted repository root in WSL, preview the two components:

```bash
python3 simulation/setup_simulation.py check --root external/linux-setup
python3 simulation/setup_simulation.py sources --root external/linux-setup
python3 simulation/setup_simulation.py install --component octo --root external/linux-setup
python3 simulation/setup_simulation.py install --component rt1x --root external/linux-setup
```

Run the source and environment setup:

```bash
python3 simulation/setup_simulation.py sources --root external/linux-setup --execute
python3 simulation/setup_simulation.py install --component octo --root external/linux-setup --execute
python3 simulation/setup_simulation.py install --component rt1x --root external/linux-setup --execute
```

Octo and RT-1-X deliberately have separate environments. Octo uses its pinned
source requirements plus the listed compatibility constraints and CUDA JAX
wheel. RT-1-X uses the dependencies required by the original TensorFlow loader.
The helper runs `pip check` and stops if a command fails; it does not silently
choose another policy version to make installation succeed.

Inspect public model identities without downloading weights:

```bash
python3 simulation/setup_simulation.py models --component octo --root external/linux-setup --metadata-only --execute
python3 simulation/setup_simulation.py models --component rt1x --root external/linux-setup --metadata-only --execute
```

Then download:

```bash
python3 simulation/setup_simulation.py models --component octo --root external/linux-setup --execute
python3 simulation/setup_simulation.py models --component rt1x --root external/linux-setup --execute
```

Octo-small/base are cached under their original `hf://rail-berkeley/...` names,
with revision checks and file hashes. The helper also fetches T5 configuration
and tokenizer assets. RT-1-X uses the fixed-generation public GCS archive,
checks its byte count and MD5 before extraction, records SHA-256 hashes, and
resolves its Universal Sentence Encoder dependency into the local TF Hub cache.
An existing Octo receipt is checked before cache reuse; changed files do not
get accepted by replacing their old hashes. The language-model download must
contain the actual local SavedModel files, not merely return a remote URL.
Weights/assets are downloaded from their upstream publishers, not redistributed
in this repository; their licenses and availability remain upstream.

Interrupted RT-1-X transfers resume their `.part` file when the server supports
byte ranges. HF downloads use the upstream cache/resume mechanism. A corrupt
or incompatible existing file causes an error instead of being overwritten.
Install steps may be repeated only for an environment created by the identical
helper recipe; if the recipe changes, use a new setup root.

Keep these cache paths when starting the original servers:

```bash
setup_root="$PWD/external/linux-setup"
export HF_HOME="$setup_root/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$setup_root/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$setup_root/cache/huggingface/hub"
export TFHUB_CACHE_DIR="$setup_root/cache/tfhub"
export ACTIVE_SIM_EVAL_SIMPLER_ROOT="$setup_root/src/SimplerEnv"
export ACTIVE_SIM_EVAL_RT1_CHECKPOINT="$setup_root/models/rt_1_x_tf_trained_for_002272480_step"
```

For Octo-small, start one server with:

```bash
ACTIVE_SIM_EVAL_OCTO_CHECKPOINT=hf://rail-berkeley/octo-small \
  "$setup_root/envs/octo/bin/python" simulation/octo-small-policy-server.py
```

For Octo-base, change the checkpoint to `hf://rail-berkeley/octo-base`. For
RT-1-X, use `"$setup_root/envs/rt1x/bin/python" simulation/rt1-x-policy-server.py`.
Do not run multiple servers on port 8765. They bind to `0.0.0.0`; keep the port
on a trusted local network and do not expose it to the internet. The helper
does not configure networking or firewalls.

## C. One episode, then the desired study

Wait for the policy server's `ready` message, then use the Windows simulator
terminal to follow [the one-episode example](README.md#try-one-episode).
Continue to the complete scenario set only after that output is reasonable.
Use a new acquisition specification with the RT-1-X checkpoint path above;
never rewrite raw observations to match a historical local path. The separate
[12-seed study](ENVIRONMENT_SEEDS.md) retains an additional historical path
requirement documented there.

`python3 simulation/setup_simulation.py settings --root external/linux-setup`
(or `python ... --root external/windows-setup` in PowerShell) prints paths
without changing the shell. No helper step changes the numerical algorithms,
scenario inputs, supplied outcomes, or paper reference results.

## Source references

- [SIMPLER ManiSkill3 source and instructions](https://github.com/simpler-env/SimplerEnv/tree/52a5088ca4bfc3a7159828af08874a198d6f95c3)
- [Octo 1.0 source requirements](https://github.com/octo-models/octo/tree/653c54acde686fde619855f2eac0dd6edad7116b)
- [ManiSkill installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [Octo-small](https://huggingface.co/rail-berkeley/octo-small), [Octo-base](https://huggingface.co/rail-berkeley/octo-base)
