from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RankSplitSourceImportTests(unittest.TestCase):
    def test_package_import_from_repository_root(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from scripts.fixed_preference import BetaPosterior; "
                    "from scripts.ranksplit import boundary_clarity, ranksplit_score; "
                    "print(BetaPosterior.__name__, boundary_clarity.__name__, ranksplit_score.__name__)"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            completed.stdout.strip(),
            "BetaPosterior boundary_clarity ranksplit_score",
        )

    def test_script_directory_import_remains_supported(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.path.insert(0, 'scripts'); "
                    "from fixed_preference import BetaPosterior; "
                    "from ranksplit import ranksplit_score; "
                    "print(BetaPosterior.__name__, ranksplit_score.__name__)"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(completed.stdout.strip(), "BetaPosterior ranksplit_score")


if __name__ == "__main__":
    unittest.main()
