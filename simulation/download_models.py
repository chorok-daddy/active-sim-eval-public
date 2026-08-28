"""Download optional public checkpoints into an explicit private setup cache.

Called explicitly by setup_simulation.py. No policy inference or simulation.
Hugging Face and TensorFlow Hub perform their normal cache management. RT-1-X
uses a pinned public GCS object, checked before safe extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

from setup_simulation import checked_root, environment, write_json


OCTO_REVISIONS = {
    "rail-berkeley/octo-small": "03d88976c54a58e10480d2043a8c762b35bc2611",
    "rail-berkeley/octo-base": "39d6c88fdbbcf6f841481a7d732f68c612d04609",
    "t5-base": "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
}
RT1_NAME = "rt_1_x_tf_trained_for_002272480_step"
RT1_URL = ("https://storage.googleapis.com/gdm-robotics-open-x-embodiment/"
           "open_x_embodiment_and_rt_x_oss/" + RT1_NAME + ".zip?generation=1696960741689538")
RT1_BYTES = 774610722
RT1_MD5 = "02218fb496258a6a37a8e14d19b5996c"
USE_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_file(url, destination, expected_size, expected_digest, opener=urllib.request.urlopen, algorithm="md5"):
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.is_symlink() or partial.is_symlink():
        raise ValueError("download destination is a symlink")
    if destination.exists():
        if destination.stat().st_size != expected_size or file_hash(destination, algorithm) != expected_digest:
            raise ValueError("existing download differs; refusing overwrite")
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise ValueError("partial download is larger than the pinned object")
    if offset < expected_size:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with opener(request, timeout=60) as response:
            status = response.getcode()
            if status == 206:
                if response.headers.get("Content-Range") != f"bytes {offset}-{expected_size-1}/{expected_size}":
                    raise ValueError("server returned a different byte range")
                mode = "ab"
            elif status == 200:
                mode, offset = "wb", 0  # Server ignored Range; restart only this owned .part file.
            else:
                raise ValueError(f"unexpected download status {status}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with partial.open(mode) as out:
                received, last_report = offset, offset
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    received += len(block)
                    if received > expected_size:
                        raise ValueError("download exceeded pinned size")
                    out.write(block)
                    if received - last_report >= 64 * 1024 * 1024:
                        print(f"{destination.name}: {received}/{expected_size} bytes", flush=True)
                        last_report = received
    if partial.stat().st_size != expected_size or file_hash(partial, algorithm) != expected_digest:
        raise ValueError("incomplete or corrupt download; .part retained, nothing extracted")
    partial.replace(destination)


def extract_directory_archive(archive_path, destination, expected_root=None, required_files=()):
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination already exists; refusing to overwrite")
    with zipfile.ZipFile(archive_path) as archive:
        seen, roots = set(), set()
        reserved = {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)], *[f"lpt{i}" for i in range(1, 10)]}
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if (not parts or parts[0] == "/" or ".." in parts or "\\" in member.filename
                or any(":" in p or p.endswith((".", " ")) for p in parts)
                or any(p.split(".")[0].casefold() in reserved for p in parts)
                or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("unsafe ZIP member")
            roots.add(parts[0])
            key = member.filename.rstrip("/").casefold()
            if key in seen:
                raise ValueError("duplicate ZIP member")
            seen.add(key)
        if len(roots) != 1 or (expected_root is not None and roots != {expected_root}):
            raise ValueError("unexpected archive root")
        archive_root = next(iter(roots))
        total = sum(i.file_size for i in archive.infolist())
        if total > 8 * 1024**3:
            raise ValueError("unexpectedly large extracted archive")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(destination.parent).free < total + 512 * 1024**2:
            raise ValueError("insufficient disk space for extraction")
        staging = Path(tempfile.mkdtemp(prefix="archive-extract-", dir=destination.parent))
        try:
            archive.extractall(staging)
            extracted = staging / archive_root
            if not extracted.is_dir() or any(not (extracted / name).is_file() for name in required_files):
                raise ValueError("archive does not contain the expected files")
            # In particular, do not create destination before renaming on Windows.
            if destination.exists() or destination.is_symlink():
                raise ValueError("destination appeared during extraction; refusing overwrite")
            extracted.rename(destination)
        finally:
            shutil.rmtree(staging)  # Only the new, private staging directory made above.


def extract_checkpoint(archive_path, models_dir):
    extract_directory_archive(archive_path, models_dir / RT1_NAME, RT1_NAME,
                              ("saved_model.pb", "variables/variables.index"))


def cache_manifest(directory, cache_root, required_files=()):
    directory, cache_root = Path(directory), Path(cache_root).resolve()
    if not directory.is_dir() or not directory.resolve().is_relative_to(cache_root):
        raise ValueError("model resolver did not return a directory inside the configured cache")
    if any(not (directory / name).is_file() for name in required_files):
        raise ValueError("cached model is missing required files")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            if not path.resolve().is_relative_to(cache_root):
                raise ValueError("cached model file points outside the configured cache")
            files[str(path.relative_to(directory))] = file_hash(path)
    if not files:
        raise ValueError("cached model contains no files")
    return files


def verify_octo_receipt(root, receipt):
    if set(receipt) != set(OCTO_REVISIONS):
        raise ValueError("existing Octo receipt has different models")
    for repo, expected_revision in OCTO_REVISIONS.items():
        entry = receipt[repo]
        if entry.get("revision") != expected_revision:
            raise ValueError("existing Octo receipt has a different revision")
        actual = cache_manifest(entry["path"], root / "cache/huggingface/hub")
        if actual != entry.get("files"):
            raise ValueError("existing Octo cache differs from its receipt; preserving both")


def octo(root, metadata_only):
    from huggingface_hub import HfApi, snapshot_download
    receipt_path = root / "receipts/models-octo.json"
    previous = None
    if not metadata_only and receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        verify_octo_receipt(root, previous)
    api, snapshots = HfApi(token=False), {}
    for repo, revision in OCTO_REVISIONS.items():
        info = api.model_info(repo, revision="main")
        if info.sha != revision:
            raise ValueError(f"upstream {repo} changed revision; inspect before using different weights")
        print(json.dumps({"model": repo, "revision": revision, "metadata_only": metadata_only}), flush=True)
        if metadata_only:
            continue
        # Use the original repo ID and main ref so original hf:// names work
        # from this cache. Pre/post checks bind the resolved revision.
        patterns = ["config.json", "spiece.model", "tokenizer*", "special_tokens_map.json"] if repo == "t5-base" else None
        snapshot = Path(snapshot_download(repo, revision="main", token=False,
                                         cache_dir=str(root / "cache/huggingface/hub"),
                                         allow_patterns=patterns, resume_download=True, max_workers=4))
        if snapshot.name != revision:
            raise ValueError("model revision changed during download")
        snapshots[repo] = {"revision": revision, "path": str(snapshot),
                           "files": cache_manifest(snapshot, root / "cache/huggingface/hub")}
    if not metadata_only:
        if previous is not None and snapshots != previous:
            raise ValueError("Octo download differs from the existing receipt; preserving receipt")
        if previous is None:
            write_json(receipt_path, snapshots)


def resolve_language_model(root, hub, previous=None):
    cache_root = root / "cache/tfhub"
    required = ("saved_model.pb", "variables/variables.index")
    if previous is not None:
        actual = cache_manifest(previous["cache_path"], cache_root, required)
        if actual != previous.get("files"):
            raise ValueError("existing language-model cache differs from its receipt")
    # UNCOMPRESSED may return a remote gs:// path, not a completed download.
    os.environ.pop("TFHUB_MODEL_LOAD_FORMAT", None)
    cache_path = hub.resolve(USE_URL)
    files = cache_manifest(cache_path, cache_root, required)
    result = {"url": USE_URL, "cache_path": str(cache_path), "files": files}
    if previous is not None and result != previous:
        raise ValueError("language-model download differs from its receipt")
    return result


def rt1x(root, metadata_only):
    with urllib.request.urlopen(urllib.request.Request(RT1_URL, method="HEAD"), timeout=30) as response:
        if int(response.headers.get("Content-Length", -1)) != RT1_BYTES:
            raise ValueError("RT-1-X remote size differs from the inspected public object")
        print(json.dumps({"model": RT1_NAME, "bytes": RT1_BYTES, "metadata_only": metadata_only}), flush=True)
    if metadata_only:
        return
    archive_path = root / "downloads" / (RT1_NAME + ".zip")
    fetch_file(RT1_URL, archive_path, RT1_BYTES, RT1_MD5)
    destination = root / "models" / RT1_NAME
    receipt_path = root / "receipts/models-rt1x.json"
    if destination.exists():
        if not receipt_path.exists():
            raise ValueError("existing checkpoint has no matching download receipt; refusing overwrite")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("archive_sha256") != file_hash(archive_path):
            raise ValueError("checkpoint receipt identifies a different archive")
        for relative, digest in receipt["files"].items():
            if file_hash(destination / relative) != digest:
                raise ValueError("existing checkpoint file differs")
    else:
        extract_checkpoint(archive_path, root / "models")
        receipt = {"url": RT1_URL, "archive_sha256": file_hash(archive_path), "files": {
            str(p.relative_to(destination)): file_hash(p) for p in sorted(destination.rglob("*")) if p.is_file()}}
        write_json(receipt_path, receipt)
    # The original RT1Inference loader also needs this public language model.
    # resolve downloads to the configured cache without loading it for inference.
    import tensorflow_hub as hub
    receipt["language_model"] = resolve_language_model(root, hub, receipt.get("language_model"))
    write_json(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("octo", "rt1x"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--execute", action="store_true", help="permit network access and downloads")
    args = parser.parse_args()
    root = checked_root(args.root)
    if not args.execute:
        print(json.dumps({"component": args.component, "root": str(root),
                          "metadata_only": args.metadata_only, "execute": False}))
        print("Preview only. Add --execute for metadata access or downloads.")
        return
    os.environ.update(environment(root))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.component == "octo":
        octo(root, args.metadata_only)
    else:
        rt1x(root, args.metadata_only)


if __name__ == "__main__":
    main()
