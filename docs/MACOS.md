# Start on macOS

[Windows](WINDOWS.md) · [macOS](MACOS.md)

## Stage 1: reproduce the paper results (CPU)

### Check the source and reconstruct the results

Extract the source ZIP and open **Terminal** in the extracted directory. With
CPython 3.11.x, run:

```bash
python3 --version
python3 scripts/verify_source.py
python3 scripts/reproduce_paper.py --mode stored
python3 scripts/reproduce_environment_seeds.py --mode stored
```

No package installation, GPU, simulator, or model download is needed. If your
`python3` points to another version, use your `python3.11` executable instead.
The first check ends with `PUBLIC SOURCE VERIFY PASS`. The next two commands
rebuild every study's statistics from the supplied individual results; the
seed command also rebuilds its tensor from original observations. Output
goes into `rerun/`, leaving the supplied data unchanged.

These are **stored reconstructions**, not newly simulated robot trials.

### Execute the allocation algorithms again

Start with two repetitions on the recorded outcomes:

```bash
python3 scripts/reproduce_paper.py --mode replay --study ranking --repetitions 2 --workers 2 --output rerun/macos-smoke
```

Then execute the preference, ranking-comparator, and environment-seed studies:

```bash
python3 scripts/reproduce_paper.py --mode replay --study preference --workers 4 --output rerun/macos-preference
python3 scripts/reproduce_paper.py --mode replay --study ranking --workers 4 --output rerun/macos-ranking
python3 scripts/reproduce_environment_seeds.py --mode replay --workers 4 --output rerun/macos-seeds
```

Repeat the same command with `--resume` if interrupted. These fresh paths
were validated on macOS arm64 CPython 3.11.3. Exact mechanism replay was
validated on Windows x86-64 CPython 3.11.15. The same source executes on both,
but nearly tied EIG scores can select different trials across math libraries.
The [reproduction guide](REPRODUCTION.md#re-run-the-allocation-algorithms)
records the exact coverage and differences; a mismatch is never silently
converted into a pass. Stored reconstruction works without changing runtime.

## Stage 2: optionally generate new robot outcomes on a GPU machine

The numerical reproduction above runs locally on the Mac. The supplied
**full simulation setup is not a native macOS/MPS recipe**: its recorded
policy environments use NVIDIA CUDA, with native Windows simulation and
WSL2 Linux inference.

If you also have access to such a machine:

1. Download and extract the same source ZIP on that machine.
2. Follow the [Windows simulation entry](WINDOWS.md#stage-2-optionally-generate-new-robot-outcomes)
   and [setup helper](../simulation/SETUP.md). An ordinary remote desktop or
   terminal is sufficient; no special controller is required.
3. Run one episode first, then the desired scenario sets. Keep new raw
   outputs and their acquisition specification separate from the paper data.
4. Copy those files back to the Mac if desired and follow
   [raw outcomes to tensor and replay](../simulation/README.md#build-a-tensor-from-new-outcomes).
   This conversion and analysis is CPU-only.

Access to a GPU machine is optional: all supplied paper-result evidence and
allocation source can be inspected and reproduced locally. Native Mac full
simulation is not presented as tested support, and new simulator outcomes
are not required to be byte-identical to the original observations.
