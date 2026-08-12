import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "demo_suite.json"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "bl_rctn.cli", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class CLITests(unittest.TestCase):
    def scratch(self):
        root = PROJECT_ROOT / "runs" / ".tmp"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def test_demo_builds_complete_verified_run(self):
        with self.scratch() as raw:
            run_dir = Path(raw) / "demo"
            completed = run_cli(
                "demo",
                "--config",
                str(CONFIG),
                "--run-dir",
                str(run_dir),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["case_count"], 18)
            self.assertTrue(manifest["all_verified"])
            with (run_dir / "summary.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = tuple(csv.DictReader(stream))
            self.assertEqual(len(rows), 18)
            self.assertTrue(all(row["verified"] == "true" for row in rows))

    def test_individual_stages_are_write_once_and_reauditable(self):
        with self.scratch() as raw:
            root = Path(raw)
            cases = root / "cases.jsonl"
            run_dir = root / "staged"
            commands = (
                (
                    "generate-samples",
                    "--config",
                    str(CONFIG),
                    "--output",
                    str(cases),
                ),
                (
                    "run",
                    "--input",
                    str(cases),
                    "--run-dir",
                    str(run_dir),
                    "--config",
                    str(CONFIG),
                ),
                ("verify", "--run-dir", str(run_dir)),
                ("export-gpt", "--run-dir", str(run_dir)),
                ("verify", "--run-dir", str(run_dir)),
            )
            for command in commands:
                completed = run_cli(*command)
                self.assertEqual(
                    completed.returncode,
                    0,
                    f"{command}: {completed.stderr}",
                )
            self.assertEqual(
                len(tuple((run_dir / "engine").glob("*.json"))), 18
            )
            self.assertEqual(
                len(tuple((run_dir / "verification").glob("*.json"))), 18
            )

    def test_unsafe_paths_fail_before_target_access(self):
        outside = "/private/tmp/bl-rctn-outside-sentinel"
        for run_path in ("../escape", outside):
            with self.subTest(path=run_path):
                completed = run_cli(
                    "demo",
                    "--config",
                    str(CONFIG),
                    "--run-dir",
                    run_path,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("algorithm", completed.stderr)

        with self.scratch() as raw:
            link = Path(raw) / "link"
            link.symlink_to("../outside-sentinel", target_is_directory=True)
            completed = run_cli(
                "demo",
                "--config",
                str(CONFIG),
                "--run-dir",
                str(link / "demo"),
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("symlink", completed.stderr.lower())


if __name__ == "__main__":
    unittest.main()
