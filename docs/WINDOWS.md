# Start on Windows

[Windows](WINDOWS.md) · [macOS](MACOS.md)

## Stage 1: reproduce the paper results (CPU)

### Check the source and reconstruct the results

Extract the source ZIP, open **PowerShell** in the extracted directory, and
use CPython 3.11.x. No package installation, GPU, model download, or WSL is
needed for these commands:

```powershell
python --version
python scripts/verify_source.py
python scripts/reproduce_paper.py --mode stored
python scripts/reproduce_environment_seeds.py --mode stored
```

If `python` is not available but the Python Launcher is installed, replace
`python` with `py -3.11`. Confirm that the version is 3.11 before continuing.
The first check ends with `PUBLIC SOURCE VERIFY PASS`. The next two commands
rebuild the statistics of all studies from their supplied individual results;
the seed command also rebuilds its tensor from original observations. Output
goes into `rerun/`; the supplied paper data are not changed.

These are **stored reconstructions**, not newly simulated robot trials.

### Execute the allocation algorithms again

Start with a two-repetition check on the recorded outcomes:

```powershell
python scripts/reproduce_paper.py --mode replay --study mechanism --repetitions 2 --workers 2 --output rerun/windows-smoke
```

For all 200 mechanism-study repetitions:

```powershell
python scripts/reproduce_paper.py --mode replay --study mechanism --workers 4 --output rerun/windows-mechanism
```

Repeat the same command with `--resume` if interrupted. Use the
[reproduction guide](REPRODUCTION.md#re-run-the-allocation-algorithms) for the
other studies and longer trajectories. All use the same CPU source on both
operating systems.

Exact fresh mechanism replay was validated on Windows x86-64 CPython 3.11.15.
Later studies have an exact compatible macOS runtime; a few nearly tied EIG
choices differ across platform math libraries. The guide records those
differences. A mismatch is reported, not normalized or hidden; the stored
reconstruction above remains available on either platform.

## Stage 2: optionally generate new robot outcomes

The recorded simulation arrangement is **native Windows for ManiSkill3/SIMPLER
and WSL2 Linux with an NVIDIA GPU for policy inference**. WSL is not used to
render the simulator. This is a separate, substantially larger setup than
the CPU reproduction above.

The [setup helper guide](../simulation/SETUP.md) provides PowerShell and WSL
commands for pinned source checkout, separate environments, simulator assets,
Octo and RT-1-X weights, and their language-model dependencies. It previews
commands first; only `--execute` installs or downloads anything. It does not
change an existing unrecorded environment, start a server, or launch a batch.

After setup, follow [one episode, then a scenario set](../simulation/README.md#try-one-episode).
The original simulator code is supplied, but the new installation recipe
has not been certified by a fresh end-to-end GPU rollout. A successful CPU
check is not a GPU compatibility check.
