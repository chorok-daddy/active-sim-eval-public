"""Fetch the two required official assets without overwriting existing folders.

Unlike the upstream 3.0.1 ZIP helper, extraction renames a verified temporary
directory to an absent destination, which also works with Windows rename rules.
No simulator environment is instantiated and no episode is run.
"""

from __future__ import annotations

import argparse
import json
import os

from download_models import extract_directory_archive, fetch_file
from setup_simulation import TASKS, checked_root, environment, write_json


ASSETS = {
    "bridge_v2_real2sim": {
        "url": "https://huggingface.co/datasets/haosulab/ManiSkill_bridge_v2_real2sim/resolve/main/bridge_v2_real2sim_dataset.zip",
        "bytes": 81962479,
        "sha256": "618512a205b4528cafecdad14b1788ed1130879f3064deb406516ed5b9c5ba92",
        "upstream_checksum": "618512a205b4528cafecdad14b1788ed1130879f3064deb406516ed5b9c5ba92",
        "target": "tasks/bridge_v2_real2sim_dataset",
    },
    "widowx250s": {
        "url": "https://github.com/haosulab/ManiSkill-WidowX250S/archive/refs/tags/v0.2.0.zip",
        "bytes": 313268,
        "sha256": "e0f9e4f35e976e3adbdb11da3e2cf04525d7dccecc88a28aba65c61de0ac88ce",
        "upstream_checksum": None,
        "target": "robots/widowx",
    },
}


def check_upstream():
    import mani_skill.envs  # registers task groups; no environment is instantiated
    from mani_skill.utils.assets import expand_data_group_into_individual_data_source_ids
    from mani_skill.utils.assets.data import DATA_SOURCES
    ids = list(dict.fromkeys([uid for task in TASKS
                             for uid in expand_data_group_into_individual_data_source_ids(task)] + ["widowx250s"]))
    if set(ids) != set(ASSETS):
        raise ValueError("installed simulator uses different task/robot assets")
    for uid, expected in ASSETS.items():
        source = DATA_SOURCES[uid]
        if (source.url != expected["url"] or source.target_path != expected["target"]
                or source.checksum != expected["upstream_checksum"]):
            raise ValueError(f"installed simulator asset definition changed: {uid}")


def download(root):
    root = checked_root(root)
    # Check all targets before downloading or extracting any of them.
    for spec in ASSETS.values():
        destination = root / "assets" / spec["target"]
        if any(p.is_symlink() for p in (destination, *destination.parents)):
            raise ValueError("asset destination contains a symlink")
        if destination.exists():
            raise ValueError("existing assets preserved; refusing overwrite")
    for uid, spec in ASSETS.items():
        archive = root / "downloads" / f"{uid}.zip"
        fetch_file(spec["url"], archive, spec["bytes"], spec["sha256"], algorithm="sha256")
        extract_directory_archive(archive, root / "assets" / spec["target"])
    write_json(root / "receipts/assets.json", {"assets": ASSETS, "gpu_execution": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = checked_root(args.root)
    print(json.dumps({"assets": ASSETS, "root": str(root), "execute": args.execute}, indent=2))
    if not args.execute:
        print("Preview only. Add --execute to download assets.")
        return
    os.environ.update(environment(root))
    check_upstream()
    download(root)


if __name__ == "__main__":
    main()
