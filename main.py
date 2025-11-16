#!/usr/bin/env python3
"""RenaML - Main Entry Point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agno_team import RenaMLTeam


def main() -> None:
    print("=" * 80)
    print("RenaML - Automated Machine Learning Pipeline")
    print("=" * 80)
    print()

    data_path = "data/input.csv"
    target_column = "species"
    task_type = "classification"

    if not Path(data_path).exists():
        print(f"❌ Error: Data file not found at '{data_path}'")
        print("Please ensure the input data file exists.")
        sys.exit(1)

    print(f"📊 Input Data: {data_path}")
    print(f"🎯 Target Column: {target_column}")
    print(f"🔬 Task Type: {task_type}")
    print()

    team = RenaMLTeam()

    results = team.run_pipeline(
        data_path=data_path,
        target_column=target_column,
        task_type=task_type,
        top_k_features=20,
        n_models=5,
    )

    if results["status"] == "success":
        print("\n✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
