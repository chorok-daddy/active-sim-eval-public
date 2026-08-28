"""Offline setup safety tests: no package installation, network, or GPU use."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile


SIMULATION = Path(__file__).resolve().parents[1] / "simulation"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SIMULATION / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


setup = load("setup_simulation", "setup_simulation.py")
downloads = load("download_models", "download_models.py")
assets = load("download_assets", "download_assets.py")


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}

    def getcode(self):
        return self.status


class SetupTests(unittest.TestCase):
    def test_preview_never_runs_or_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new-setup"
            for arguments in (("sources",), ("install", "--component", "simulator"),
                              ("install", "--component", "octo"), ("install", "--component", "rt1x"),
                              ("models", "--component", "octo"), ("assets",), ("settings",), ("check",)):
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(setup,"run") as runner:
                    self.assertEqual(setup.main([*arguments,"--root",str(root)]),0)
                    runner.assert_not_called()
                    self.assertFalse(root.exists())

    def test_system_and_component_paths(self):
        root = Path("example")
        self.assertEqual(setup.python_path(root,"simulator","Windows"),root/"envs/simulator/python.exe")
        self.assertEqual(setup.python_path(root,"octo","Linux"),root/"envs/octo/bin/python")
        for component, system in (("simulator","Linux"),("octo","Windows"),("rt1x","Darwin")):
            with self.assertRaises(ValueError):
                setup.require_platform(component,system)

    def test_no_system_install_or_shell_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for component in setup.PYTHONS:
                system = "Windows" if component == "simulator" else "Linux"
                commands = setup.install_commands(root,component,system,"conda")
                self.assertIn(str(root/"envs"/component),commands[0])
                for command in commands[1:]:
                    self.assertEqual(command[0],str(setup.python_path(root,component,system)))
                    self.assertNotIn("--user",command)
                self.assertEqual(commands[-1][-2:],["pip","check"])

    def test_windows_pinocchio_is_installed_from_conda(self):
        commands=setup.install_commands(Path("example"),"simulator","Windows","conda.exe")
        self.assertIn("pinocchio=3.9.0",commands[0])
        self.assertNotIn("pin==",(SIMULATION/"requirements/simulator.txt").read_text())

    def test_broad_root_and_symlink_are_rejected(self):
        for target in (Path.home(),Path.cwd(),Path(Path.cwd().anchor)):
            with self.assertRaises(ValueError):
                setup.checked_root(target)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"setup"
            root.mkdir()
            try:
                (root/"models").symlink_to(Path(temp),target_is_directory=True)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(ValueError,"symlink"):
                setup.checked_root(root)

    def test_existing_sources_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"src/SimplerEnv").mkdir(parents=True)
            with mock.patch.object(setup,"run") as runner, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    setup.sources(root,True)
                runner.assert_not_called()

    def test_different_git_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/".git").mkdir()
            with mock.patch.object(setup,"git_value",side_effect=["url","different"]):
                with self.assertRaisesRegex(ValueError,"revision"):
                    setup.verify_source(root,"url","expected")

    def test_existing_assets_are_not_passed_to_upstream_deleter(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"assets").mkdir()
            with mock.patch.object(setup,"run") as runner, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"overwrite"):
                    setup.assets(root,"Windows",True)
                runner.assert_not_called()

    def test_asset_plan_has_four_tasks_and_deduplicates_shared_ids(self):
        command=setup.asset_commands(Path("example"),"Windows")
        self.assertTrue(command[1].endswith("download_assets.py"))
        self.assertEqual(len(setup.TASKS),4)
        self.assertEqual(set(assets.ASSETS),{"bridge_v2_real2sim","widowx250s"})
        self.assertEqual(len({s["target"] for s in assets.ASSETS.values()}),2)

    def test_unrecorded_environment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/"envs/octo").mkdir(parents=True)
            with mock.patch.object(setup,"verify_source"), mock.patch.object(setup.shutil,"which",return_value="conda"), mock.patch.object(setup,"run") as runner, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"unrecorded"):
                    setup.install(root,"octo","Linux","conda",True)
                runner.assert_not_called()

    def test_failed_commands_do_not_continue(self):
        with mock.patch.object(setup.subprocess,"run",side_effect=subprocess.CalledProcessError(1,["test"])), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                setup.run(["test"])


class DownloadTests(unittest.TestCase):
    def test_sha256_download_verification(self):
        data=b"asset archive"
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/"asset.zip"
            downloads.fetch_file("https://example.invalid",target,len(data),hashlib.sha256(data).hexdigest(),
                                 lambda *a,**k:Response(data),algorithm="sha256")
            self.assertEqual(target.read_bytes(),data)

    def test_windows_rename_requires_absent_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive=root/"asset.zip"
            with zipfile.ZipFile(archive,"w") as out:
                out.writestr("Upstream-0.2.0/robot.urdf",b"official asset")
            destination=root/"assets/robots/widowx"
            rename=Path.rename
            def windows_rename(source,target):
                self.assertFalse(Path(target).exists())
                return rename(source,target)
            with mock.patch.object(Path,"rename",new=windows_rename):
                downloads.extract_directory_archive(archive,destination)
            self.assertEqual((destination/"robot.urdf").read_bytes(),b"official asset")
            with self.assertRaisesRegex(ValueError,"overwrite"):
                downloads.extract_directory_archive(archive,destination)

    def test_existing_assets_are_preserved_before_any_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            destination=root/"assets/robots/widowx"
            destination.mkdir(parents=True)
            (destination/"keep").write_bytes(b"existing")
            with mock.patch.object(assets,"fetch_file") as fetch:
                with self.assertRaisesRegex(ValueError,"overwrite"):
                    assets.download(root)
                fetch.assert_not_called()
            self.assertEqual((destination/"keep").read_bytes(),b"existing")

    def test_asset_downloader_preview_never_imports_simulator(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"new"
            with mock.patch.object(assets,"check_upstream") as upstream, mock.patch.object(assets,"download") as fetch, contextlib.redirect_stdout(io.StringIO()):
                assets.main(["--root",str(root)])
                upstream.assert_not_called()
                fetch.assert_not_called()
                self.assertFalse(root.exists())

    def test_zip_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive=root/"bad.zip"
            link=zipfile.ZipInfo("root/link")
            link.create_system=3
            link.external_attr=0o120777 << 16
            with zipfile.ZipFile(archive,"w") as out:
                out.writestr(link,"../../outside")
            with self.assertRaisesRegex(ValueError,"unsafe"):
                downloads.extract_directory_archive(archive,root/"output")

    def test_changed_octo_cache_preserves_original_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            snapshots={}
            for repo,revision in downloads.OCTO_REVISIONS.items():
                path=root/"cache/huggingface/hub"/("models--"+repo.replace("/","--"))/"snapshots"/revision
                path.mkdir(parents=True)
                (path/"config.json").write_bytes(b"original")
                snapshots[repo]=str(path)
            client=mock.Mock()
            client.model_info.side_effect=lambda repo,**kwargs: types.SimpleNamespace(sha=downloads.OCTO_REVISIONS[repo])
            module=types.SimpleNamespace(HfApi=mock.Mock(return_value=client),
                snapshot_download=mock.Mock(side_effect=lambda repo,**kwargs:snapshots[repo]))
            with mock.patch.dict(sys.modules,{"huggingface_hub":module}), contextlib.redirect_stdout(io.StringIO()):
                downloads.octo(root,False)
                receipt=root/"receipts/models-octo.json"
                before=receipt.read_bytes()
                downloads.octo(root,False)
                self.assertEqual(receipt.read_bytes(),before)
                changed=Path(snapshots["rail-berkeley/octo-small"])/"config.json"
                changed.write_bytes(b"changed")
                calls=module.snapshot_download.call_count
                with self.assertRaisesRegex(ValueError,"differs"):
                    downloads.octo(root,False)
                self.assertEqual(receipt.read_bytes(),before)
                self.assertEqual(changed.read_bytes(),b"changed")
                self.assertEqual(module.snapshot_download.call_count,calls)

    def test_changed_upstream_octo_revision_stops_before_download(self):
        with tempfile.TemporaryDirectory() as temp:
            client=mock.Mock()
            client.model_info.return_value=types.SimpleNamespace(sha="different")
            module=types.SimpleNamespace(HfApi=mock.Mock(return_value=client),snapshot_download=mock.Mock())
            with mock.patch.dict(sys.modules,{"huggingface_hub":module}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,"changed revision"):
                    downloads.octo(Path(temp),False)
                module.snapshot_download.assert_not_called()

    def test_language_model_remote_path_is_not_a_download(self):
        with tempfile.TemporaryDirectory() as temp:
            def remote(url):
                self.assertNotIn("TFHUB_MODEL_LOAD_FORMAT",setup.os.environ)
                return "gs://remote/model"
            hub=types.SimpleNamespace(resolve=mock.Mock(side_effect=remote))
            with mock.patch.dict(setup.os.environ,{"TFHUB_MODEL_LOAD_FORMAT":"UNCOMPRESSED"}):
                with self.assertRaisesRegex(ValueError,"configured cache"):
                    downloads.resolve_language_model(Path(temp),hub)

    def test_language_model_requires_complete_local_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            cache=root/"cache/tfhub/model"
            cache.mkdir(parents=True)
            (cache/"saved_model.pb").write_bytes(b"model")
            hub=types.SimpleNamespace(resolve=mock.Mock(return_value=str(cache)))
            with self.assertRaisesRegex(ValueError,"missing"):
                downloads.resolve_language_model(root,hub)
            (cache/"variables").mkdir()
            (cache/"variables/variables.index").write_bytes(b"index")
            record=downloads.resolve_language_model(root,hub)
            self.assertEqual(len(record["files"]),2)
            (cache/"saved_model.pb").write_bytes(b"corrupt")
            calls=hub.resolve.call_count
            with self.assertRaisesRegex(ValueError,"differs"):
                downloads.resolve_language_model(root,hub,record)
            self.assertEqual(hub.resolve.call_count,calls)

    def test_direct_downloader_is_preview_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"new-setup"
            with mock.patch.object(sys,"argv",["download_models.py","--component","rt1x","--root",str(root)]), mock.patch.object(downloads,"rt1x") as runner, contextlib.redirect_stdout(io.StringIO()):
                downloads.main()
                runner.assert_not_called()
                self.assertFalse(root.exists())

    def test_symlink_partial_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original=root/"original"
            original.write_bytes(b"preserve me")
            target=root/"test.zip"
            try:
                target.with_suffix(".zip.part").symlink_to(original)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(ValueError,"symlink"):
                downloads.fetch_file("https://example.invalid",target,3,"unused")
            self.assertEqual(original.read_bytes(),b"preserve me")

    def test_complete_download_and_verified_reuse(self):
        data=b"test bytes"
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/"test.zip"
            opener=mock.Mock(return_value=Response(data))
            downloads.fetch_file("https://example.invalid/model",target,len(data),hashlib.md5(data).hexdigest(),opener)
            self.assertEqual(target.read_bytes(),data)
            downloads.fetch_file("https://example.invalid/model",target,len(data),hashlib.md5(data).hexdigest(),opener)
            self.assertEqual(opener.call_count,1)

    def test_resume_requires_correct_range(self):
        data=b"abcdefgh"
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/"model.zip"
            target.with_suffix(".zip.part").write_bytes(data[:3])
            def opener(request,timeout):
                self.assertEqual(request.get_header("Range"),"bytes=3-")
                return Response(data[3:],206,{"Content-Range":"bytes 3-7/8"})
            downloads.fetch_file("https://example.invalid/model",target,8,hashlib.md5(data).hexdigest(),opener)
            self.assertEqual(target.read_bytes(),data)

    def test_ignored_range_restarts_only_partial_file(self):
        data=b"abcdefgh"
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/"model.zip"
            target.with_suffix(".zip.part").write_bytes(b"abc")
            downloads.fetch_file("https://example.invalid/model",target,8,hashlib.md5(data).hexdigest(),lambda *a,**k:Response(data))
            self.assertEqual(target.read_bytes(),data)

    def test_corrupt_truncated_and_wrong_range_downloads_fail(self):
        for data,status,headers in ((b"bad",200,{}),(b"abc",206,{"Content-Range":"bytes 4-6/8"})):
            with tempfile.TemporaryDirectory() as temp:
                target=Path(temp)/"model.zip"
                with self.assertRaises(ValueError):
                    downloads.fetch_file("https://example.invalid/model",target,8,hashlib.md5(b"abcdefgh").hexdigest(),lambda *a,**k:Response(data,status,headers))
                self.assertFalse(target.exists())

    def test_zip_escape_and_symlinks_are_rejected(self):
        for name in ("../outside",downloads.RT1_NAME+"/../outside",downloads.RT1_NAME+"/C:bad",downloads.RT1_NAME+"\\outside"):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                archive=root/"test.zip"
                with zipfile.ZipFile(archive,"w") as out:
                    out.writestr(name,b"x")
                with self.assertRaisesRegex(ValueError,"unsafe"):
                    downloads.extract_checkpoint(archive,root/"models")
                self.assertFalse((root/"outside").exists())

    def test_valid_minimal_checkpoint_extracts_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            archive=root/"test.zip"
            with zipfile.ZipFile(archive,"w") as out:
                out.writestr(downloads.RT1_NAME+"/saved_model.pb",b"model")
                out.writestr(downloads.RT1_NAME+"/variables/variables.index",b"index")
            downloads.extract_checkpoint(archive,root/"models")
            self.assertEqual((root/"models"/downloads.RT1_NAME/"saved_model.pb").read_bytes(),b"model")
            with self.assertRaisesRegex(ValueError,"overwrite"):
                downloads.extract_checkpoint(archive,root/"models")

    def test_text_dependencies_are_not_omitted(self):
        self.assertIn("t5-base",downloads.OCTO_REVISIONS)
        self.assertEqual(downloads.USE_URL,"https://tfhub.dev/google/universal-sentence-encoder-large/5")


if __name__ == "__main__":
    unittest.main()
