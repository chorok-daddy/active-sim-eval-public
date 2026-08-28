"""Optional setup helper. Preview by default; --execute explicitly changes files.

This is an installation/download convenience, not an experiment runner or a
claim that a fresh GPU environment has been validated. No simulator or policy
server is started. Scientific source and supplied observations are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
SOURCES = {
    "SimplerEnv": ("https://github.com/simpler-env/SimplerEnv.git",
                   "52a5088ca4bfc3a7159828af08874a198d6f95c3"),
    "octo": ("https://github.com/octo-models/octo.git",
             "653c54acde686fde619855f2eac0dd6edad7116b"),
}
PYTHONS = {"simulator": "3.11.15", "octo": "3.10.20", "rt1x": "3.10.20"}
TASKS = ("PutCarrotOnPlateInScene-v1", "PutSpoonOnTableClothInScene-v1",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v1", "PutEggplantInBasketScene-v1")


def checked_root(value):
    supplied = Path(value).expanduser().absolute()
    system_aliases = {Path("/tmp"), Path("/var")} if platform.system() == "Darwin" else set()
    if any(p.is_symlink() and p not in system_aliases for p in (supplied, *supplied.parents)):
        raise ValueError("use a direct setup directory, not a symlink")
    root = supplied.resolve()
    if root in {Path(root.anchor), Path.home().resolve(), HERE.parent, Path.cwd().resolve()}:
        raise ValueError("choose a dedicated setup subdirectory, not a home/repository/filesystem root")
    for relative in ("src", "src/SimplerEnv", "src/octo", "envs", "envs/simulator",
                     "envs/octo", "envs/rt1x", "models", "downloads", "receipts", "assets",
                     "cache", "cache/huggingface", "cache/huggingface/hub", "cache/tfhub", "cache/pip", "cache/conda"):
        if (root / relative).is_symlink():
            raise ValueError(f"setup destination is a symlink: {relative}")
    return root


def python_path(root, component, system):
    prefix = root / "envs" / component
    return prefix / ("python.exe" if system == "Windows" else "bin/python")


def environment(root):
    return {
        "HF_HOME": str(root / "cache/huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(root / "cache/huggingface/hub"),
        "TRANSFORMERS_CACHE": str(root / "cache/huggingface/hub"),
        "TFHUB_CACHE_DIR": str(root / "cache/tfhub"),
        "MS_ASSET_DIR": str(root / "assets"),
        "PIP_CACHE_DIR": str(root / "cache/pip"),
        "CONDA_PKGS_DIRS": str(root / "cache/conda"),
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "ACTIVE_SIM_EVAL_SIMPLER_ROOT": str(root / "src/SimplerEnv"),
        "ACTIVE_SIM_EVAL_RT1_CHECKPOINT": str(root / "models/rt_1_x_tf_trained_for_002272480_step"),
    }


def process_environment(root, component, system):
    env = dict(os.environ, **environment(root))
    for name in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(name, None)
    prefix = root / "envs" / component
    if system == "Windows":
        env["CONDA_PREFIX"] = str(prefix)
        env["PATH"] = os.pathsep.join([str(prefix), str(prefix / "Library/bin"),
                                      str(prefix / "Scripts"), env.get("PATH", "")])
    return env


def require_platform(component, system):
    if component == "simulator" and system != "Windows":
        raise ValueError("this simulator recipe targets native Windows; it is not a WSL rendering recipe")
    if component in ("octo", "rt1x") and system != "Linux":
        raise ValueError("policy installation/model preparation targets Linux/WSL, not native Windows or Mac")


def install_commands(root, component, system, conda):
    py = str(python_path(root, component, system))
    commands = [[conda, "create", "--yes", "--override-channels", "--channel", "conda-forge",
                 "--prefix", str(root / "envs" / component), f"python={PYTHONS[component]}", "pip"]]
    if component == "simulator":
        # The recorded Windows pin metadata comes from conda-forge Pinocchio;
        # PyPI pin 3.9.0 has no Windows wheel.
        commands[0] += ["pinocchio=3.9.0", "numpy=2.4.6"]
    pip = [py, "-m", "pip", "install"]
    if component == "octo":
        constraints = str(HERE / "requirements/octo-constraints.txt")
        commands += [pip + ["-r", str(root / "src/octo/requirements.txt"),
                            "jax[cuda12_pip]==0.4.20", "-f",
                            "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
                            "-c", constraints],
                     pip + ["--no-deps", "-e", str(root / "src/octo")]]
    else:
        commands += [pip + ["-r", str(HERE / f"requirements/{component}.txt")],
                     pip + ["--no-deps", "-e", str(root / "src/SimplerEnv")]]
    commands += [[py, "-m", "pip", "check"]]
    return commands


def run(argv, env=None):
    print("EXEC " + json.dumps(list(map(str, argv))), flush=True)
    subprocess.run(list(map(str, argv)), check=True, env=env)


def git_value(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def verify_source(directory, url, revision):
    if not (directory / ".git").is_dir():
        raise ValueError(f"existing path is not an owned dependency checkout: {directory}")
    if git_value(directory, "remote", "get-url", "origin") != url:
        raise ValueError(f"different source origin at {directory}")
    if git_value(directory, "rev-parse", "HEAD") != revision:
        raise ValueError(f"different source revision at {directory}; use a new setup root")
    if git_value(directory, "status", "--porcelain"):
        raise ValueError(f"modified dependency checkout at {directory}; refusing to overwrite")


def sources(root, execute):
    for name, (url, revision) in SOURCES.items():
        dest = root / "src" / name
        commands = [["git", "clone", "--filter=blob:none", "--no-checkout", url, str(dest)],
                    ["git", "-C", str(dest), "checkout", "--detach", revision]]
        print(json.dumps({"source": name, "revision": revision, "commands": commands}))
        if execute:
            if dest.exists():
                verify_source(dest, url, revision)
                print(f"Already verified: {dest}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                for command in commands:
                    run(command)
                verify_source(dest, url, revision)


def recipe_identity(root, component, commands):
    files = [HERE / "setup_simulation.py", *sorted((HERE / "requirements").glob("*.txt"))]
    return {"component": component, "prefix": str(root / "envs" / component),
            "commands": commands,
            "recipe_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def install(root, component, system, conda, execute):
    conda = shutil.which(conda) or conda
    commands = install_commands(root, component, system, conda)
    print(json.dumps({"component": component, "commands": commands, "environment": environment(root)}, indent=2))
    if not execute:
        return
    require_platform(component, system)
    if not shutil.which(conda) and not Path(conda).is_file():
        raise ValueError("install Miniforge/Conda first or provide --conda /path/to/conda.exe")
    if system == "Windows" and Path(conda).suffix.lower() in (".bat", ".cmd"):
        raise ValueError("pass the real conda.exe using --conda, not a .bat/.cmd wrapper")
    for name in (("SimplerEnv", "octo") if component == "octo" else ("SimplerEnv",)):
        verify_source(root / "src" / name, *SOURCES[name])
    marker = root / "receipts" / f"install-{component}.json"
    identity = recipe_identity(root, component, commands)
    prefix = root / "envs" / component
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("identity") != identity:
            raise ValueError("existing setup recipe differs; choose a new setup root")
    elif prefix.exists():
        raise ValueError("refusing to modify an existing unrecorded environment")
    write_json(marker, {"identity": identity, "status": "installing"})
    env = process_environment(root, component, system)
    # Only an environment created by this exact recipe may be resumed.
    if prefix.exists():
        if not (prefix / "conda-meta/history").is_file() or not python_path(root, component, system).is_file():
            raise ValueError("incomplete environment creation; choose a new setup root")
        commands = commands[1:]
    for command in commands:
        run(command, env)
    write_json(marker, {"identity": identity, "status": "packages-installed-not-gpu-validated"})


def asset_commands(root, system):
    py = str(python_path(root, "simulator", system))
    return [py, str(HERE / "download_assets.py"), "--root", str(root), "--execute"]


def assets(root, system, execute):
    command = asset_commands(root, system)
    print(json.dumps({"command": command, "MS_ASSET_DIR": str(root / "assets")}, indent=2))
    if execute:
        require_platform("simulator", system)
        # Upstream can delete an existing asset directory: never point it at one.
        target = root / "assets"
        if target.exists():
            raise ValueError("asset directory already exists; refusing upstream overwrite; move that folder aside before retrying")
        if not python_path(root, "simulator", system).is_file():
            raise ValueError("install the simulator environment first")
        target.mkdir(parents=True)
        run(command, process_environment(root, "simulator", system))


def models(root, component, system, execute, metadata_only=False):
    command = [str(python_path(root, component, system)), str(HERE / "download_models.py"),
               "--component", component, "--root", str(root), "--execute"]
    if metadata_only:
        command.append("--metadata-only")
    print(json.dumps({"command": command, "environment": environment(root)}, indent=2))
    if execute:
        require_platform(component, system)
        if not python_path(root, component, system).is_file():
            raise ValueError("install the matching policy environment first")
        # Download utilities never initialize a policy or allocate its GPU.
        run(command, dict(process_environment(root, component, system), CUDA_VISIBLE_DEVICES=""))


def doctor(root, system):
    parent = root
    while not parent.exists():
        parent = parent.parent
    print(json.dumps({"python": platform.python_version(), "system": system,
                     "start_guide": "docs/ONBOARDING.md",
                     "git": shutil.which("git"), "conda": os.environ.get("CONDA_EXE") or shutil.which("conda"),
                     "nvidia_smi": shutil.which("nvidia-smi"),
                     "free_gib": round(shutil.disk_usage(parent).free / 2**30, 1),
                     "root": str(root), "writes_performed": False,
                     "note": "Driver, CUDA/Vulkan, network and GPU inference are not certified by this check."}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "sources", "install", "models", "assets", "settings"))
    parser.add_argument("--root", default="external/simulator-setup")
    parser.add_argument("--component", choices=tuple(PYTHONS))
    parser.add_argument("--conda", default=os.environ.get("CONDA_EXE") or "conda")
    parser.add_argument("--execute", action="store_true", help="actually run the printed commands")
    parser.add_argument("--metadata-only", action="store_true", help="models: inspect identities, do not download weights")
    args = parser.parse_args(argv)
    root, system = checked_root(args.root), platform.system()
    if args.action in ("install", "models") and args.component is None:
        parser.error("--component is required")
    if args.action == "models" and args.component == "simulator":
        parser.error("models requires octo or rt1x")
    if args.metadata_only and args.action != "models":
        parser.error("--metadata-only is only for models")
    if args.action == "check":
        doctor(root, system)
    elif args.action == "sources":
        sources(root, args.execute)
    elif args.action == "install":
        install(root, args.component, system, args.conda, args.execute)
    elif args.action == "models":
        models(root, args.component, system, args.execute, args.metadata_only)
    elif args.action == "assets":
        assets(root, system, args.execute)
    else:
        print(json.dumps({"environment": environment(root), "python": {
            k:str(python_path(root,k,system)) for k in PYTHONS},
            "octo_checkpoint_names": ["hf://rail-berkeley/octo-small", "hf://rail-berkeley/octo-base"],
            "note": "Export these environment settings when running the original server/client. No server is launched."}, indent=2))
    if not args.execute and args.action not in ("check", "settings"):
        print("Preview only: no downloads or writes. Add --execute to run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Setup stopped: {exc}")
