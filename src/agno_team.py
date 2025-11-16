"""Main orchestrator using Agno Team pattern."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from agents import EDAAgent, FeatSelectAgent, ModelingAgent, ReportAgent, VizAgent


class RenaMLTeam:
    """Team orchestrator for the RenaML POC pipeline."""

    def __init__(self, run_id: str | None = None, out_root: Path | None = None) -> None:
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.out_root = out_root or Path("runs")
        self.run_dir = self.out_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.memory: dict[str, Any] = {}

        self._setup_logging()

        self.logger.info(f"[Team] Initialized RenaML Team with run_id: {self.run_id}")
        self.logger.info(f"[Team] Run directory: {self.run_dir}")

        self.agents = {
            "eda": EDAAgent(self.memory, self.run_dir),
            "viz": VizAgent(self.memory, self.run_dir),
            "featselect": FeatSelectAgent(self.memory, self.run_dir),
            "modeling": ModelingAgent(self.memory, self.run_dir),
            "report": ReportAgent(self.memory, self.run_dir),
        }

    def _setup_logging(self) -> None:
        log_file = self.run_dir / "pipeline.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)

    def run_pipeline(
        self,
        data_path: str,
        target_column: str,
        task_type: str = "classification",
        top_k_features: int = 20,
        n_models: int = 5,
    ) -> dict[str, Any]:
        self.logger.info("[Team] Starting pipeline execution")
        results = {}

        try:
            self.logger.info("[Team] Step 1/5: Data Ingestion and Exploratory Data Analysis")
            results["eda"] = self.agents["eda"].run(data_path)
            if results["eda"]["status"] == "error":
                raise Exception(f"EDA failed: {results['eda']['error']}")

            self.logger.info("[Team] Step 2/5: Visualizations")
            results["viz"] = self.agents["viz"].run()
            if results["viz"]["status"] == "error":
                self.logger.warning(f"Visualization failed: {results['viz']['error']}")

            self.logger.info("[Team] Step 3/5: Feature Selection")
            results["featselect"] = self.agents["featselect"].run(
                target_column=target_column, task_type=task_type, top_k=top_k_features
            )
            if results["featselect"]["status"] == "error":
                self.logger.warning(f"Feature selection failed: {results['featselect']['error']}")

            self.logger.info("[Team] Step 4/5: Model Training")
            results["modeling"] = self.agents["modeling"].run(n_models=n_models)
            if results["modeling"]["status"] == "error":
                self.logger.warning(f"Modeling failed: {results['modeling']['error']}")

            self.logger.info("[Team] Step 5/5: Report Generation")
            results["report"] = self.agents["report"].run()
            if results["report"]["status"] == "error":
                self.logger.warning(f"Report generation failed: {results['report']['error']}")

            self.logger.info("[Team] Pipeline execution complete!")
            self.logger.info(f"[Team] Results saved to: {self.run_dir}")

            self._print_summary(results)

            return {
                "status": "success",
                "run_id": self.run_id,
                "run_dir": str(self.run_dir),
                "results": results,
            }

        except Exception as e:
            self.logger.error(f"[Team] Pipeline execution failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "partial_results": results,
            }

    def _print_summary(self, results: dict[str, Any]) -> None:
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Run Directory: {self.run_dir}")
        self.logger.info("Steps Completed:")

        for step, result in results.items():
            status = result.get("status", "unknown")
            emoji = "✅" if status == "success" else "❌"
            self.logger.info(f"  {emoji} {step.capitalize()}: {status}")

        if "report" in results and results["report"]["status"] == "success":
            self.logger.info("📊 Reports Generated:")
            if results["report"].get("markdown_path"):
                self.logger.info(f"  - Markdown: {results['report']['markdown_path']}")
            if results["report"].get("html_path"):
                self.logger.info(f"  - HTML: {results['report']['html_path']}")
            if results["report"].get("docx_path"):
                self.logger.info(f"  - DOCX: {results['report']['docx_path']}")

        if "modeling" in results and results["modeling"]["status"] == "success":
            mod_results = results["modeling"].get("results", {})
            if "best_model" in mod_results:
                self.logger.info(f"🎯 Best Model: {mod_results['best_model']}")

        if "featselect" in results and results["featselect"]["status"] == "success":
            features = results["featselect"].get("selected_features", [])
            self.logger.info(f"🔍 Selected Features: {len(features)}")
            if features:
                self.logger.info(f"  Top 5: {', '.join(features[:5])}")

        self.logger.info("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="RenaML POC - Automated ML Pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="ML task type",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Custom run ID")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top features to select")
    parser.add_argument("--n-models", type=int, default=5, help="Number of models to compare")
    parser.add_argument("--out-root", type=str, default="runs", help="Output root directory")

    args = parser.parse_args()

    team = RenaMLTeam(run_id=args.run_id, out_root=Path(args.out_root))

    results = team.run_pipeline(
        data_path=args.data,
        target_column=args.target,
        task_type=args.task,
        top_k_features=args.top_k,
        n_models=args.n_models,
    )

    sys.exit(0 if results["status"] == "success" else 1)


if __name__ == "__main__":
    main()
