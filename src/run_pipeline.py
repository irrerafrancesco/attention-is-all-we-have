"""Run the full reproducible project workflow in dependency order."""

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "data_loader.py",
    "data_explore.py",
    "data_quality.py",
    "preprocess.py",
    "baseline_model.py",
    "boosted_model.py",
    "model_comparison.py",
    "simple_rule_baseline.py",
    "review_queue.py",
    "build_database.py",
    "query_database.py",
    "create_report.py",
    "explain_model.py",
]


def main():
    for script in SCRIPTS:
        print(f"\nRunning {script}...", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "src" / script)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
