"""ModelingAgent: Orchestrates model training using PyCaret with LLM interpretation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pycaret.classification import (
    compare_models,
    finalize_model,
    pull,
    save_model,
)
from pycaret.classification import setup as setup_classification
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import finalize_model as finalize_model_reg
from pycaret.regression import pull as pull_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import setup as setup_regression

from prompts.modeling_agent_prompts import AGENT_INSTRUCTIONS, INTERPRETATION_PROMPT_TEMPLATE


class ModelingAgent:
    """Agent responsible for model training and evaluation with LLM-powered insights."""

    def __init__(self, team_memory: dict[str, Any], run_dir: Path) -> None:
        self.team_memory = team_memory
        self.run_dir = run_dir
        self.logger = logging.getLogger(__name__)
        self.models_dir = run_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create Agent instance for LLM insights
        self.llm_agent = Agent(
            name="ModelTrainer",
            role="model_trainer",
            model=OpenAIChat(id="gpt-4o"),
            instructions=AGENT_INSTRUCTIONS,
        )

    def run(self, n_models: int = 5, optimize_metric: str | None = None) -> dict[str, Any]:
        self.logger.info("[ModelTrainer] Starting model training")

        try:
            # Get configuration from team memory
            dataset_path = self.team_memory.get("dataset_path")
            target_column = self.team_memory.get("target_column")
            task_type = self.team_memory.get("task_type", "classification")
            feature_list = self.team_memory.get("feature_list")

            if not dataset_path or not target_column:
                raise ValueError("Missing dataset_path or target_column in team memory")

            # Load data
            df = pd.read_parquet(dataset_path)
            self.logger.info(f"[ModelTrainer] Loaded dataset with shape {df.shape}")

            # Filter to selected features if available
            if feature_list:
                available_features = [f for f in feature_list if f in df.columns]
                columns_to_use = available_features + [target_column]
                df = df[columns_to_use]
                self.logger.info(
                    f"[ModelTrainer] Using {len(available_features)} selected features"
                )

            # Train models using PyCaret
            if task_type == "classification":
                results = self._train_classification_models(
                    df, target_column, n_models, optimize_metric
                )
            elif task_type == "regression":
                results = self._train_regression_models(
                    df, target_column, n_models, optimize_metric
                )
            else:
                raise ValueError(
                    f"Unsupported task_type: {task_type}. Must be 'classification' or 'regression'"
                )

            # Get LLM interpretation
            llm_interpretation = self._get_llm_interpretation(results, task_type, n_models)
            results["llm_interpretation"] = llm_interpretation

            # Save summary
            summary_path = self.models_dir / "models_summary.json"
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Update team memory
            self.team_memory["models_summary"] = results
            self.team_memory["models_dir"] = str(self.models_dir)
            self.team_memory["modeling_insights"] = llm_interpretation

            self.logger.info("[ModelTrainer] Model training complete")

            return {
                "status": "success",
                "results": results,
                "summary_path": str(summary_path),
                "llm_interpretation": llm_interpretation,
            }

        except Exception as e:
            self.logger.error(f"[ModelTrainer] Error during modeling: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def _get_llm_interpretation(
        self, results: dict[str, Any], task_type: str, n_models: int
    ) -> str:
        try:
            if "error" in results:
                return f"Unable to interpret results due to error: {results['error']}"

            # Extract comparison results
            comparison_results = results.get("comparison_results", [])
            if not comparison_results:
                return "No model comparison results available for interpretation."

            best_model = results.get("best_model", "Unknown")

            # Format top models summary
            top_models_summary = []
            for i, model_result in enumerate(comparison_results[:n_models], 1):
                model_name = model_result.get("Model", f"Model {i}")
                metrics_str = ", ".join(
                    [
                        f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                        for k, v in model_result.items()
                        if k != "Model"
                    ]
                )
                top_models_summary.append(f"{i}. {model_name}: {metrics_str}")

            prompt = INTERPRETATION_PROMPT_TEMPLATE.format(
                task_type=task_type,
                best_model=best_model,
                n_models=n_models,
                top_models_summary=chr(10).join(top_models_summary),
            )

            self.logger.info("[ModelTrainer] Requesting LLM interpretation of modeling results")
            response = self.llm_agent.run(input=prompt)

            if hasattr(response, "content"):
                llm_text = response.content
            else:
                llm_text = str(response)

            self.logger.info("[ModelTrainer] LLM interpretation generated")
            return llm_text

        except Exception as e:
            self.logger.error(f"[ModelTrainer] Error getting LLM interpretation: {e}")
            return f"Error generating LLM interpretation: {str(e)}"

    def _train_classification_models(
        self, df: pd.DataFrame, target: str, n_models: int, metric: str | None
    ) -> dict[str, Any]:
        try:
            self.logger.info("[ModelTrainer] Setting up PyCaret classification...")

            setup_classification(
                data=df,
                target=target,
                session_id=42,
                verbose=False,
                html=False,
                log_experiment=False,
                system_log=False,
            )

            self.logger.info(f"[ModelTrainer] Comparing classification models (top {n_models})...")

            # Compare models
            best_models = compare_models(
                n_select=n_models,
                sort=metric or "Accuracy",
                verbose=False,
            )

            # Get comparison results
            comparison_df = pull()

            # Handle single model or list of models
            if not isinstance(best_models, list):
                best_models = [best_models]

            # Save best model
            best_model = best_models[0]
            final_model = finalize_model(best_model)
            model_path = self.models_dir / "best_classification_model.pkl"
            save_model(final_model, str(model_path.with_suffix("")))

            self.logger.info(f"[ModelTrainer] Best classification model saved to {model_path}")

            # Prepare results
            results = {
                "timestamp": datetime.now().isoformat(),
                "task_type": "classification",
                "target_column": target,
                "n_models_compared": len(best_models),
                "best_model": str(type(best_model).__name__),
                "optimize_metric": metric or "Accuracy",
                "model_path": str(model_path),
                "comparison_results": comparison_df.to_dict("records"),
            }

            return results

        except Exception as e:
            self.logger.error(f"[ModelTrainer] Error in classification training: {e}", exc_info=True)
            return {"error": str(e)}

    def _train_regression_models(
        self, df: pd.DataFrame, target: str, n_models: int, metric: str | None
    ) -> dict[str, Any]:
        try:
            self.logger.info("[ModelTrainer] Setting up PyCaret regression...")

            setup_regression(
                data=df,
                target=target,
                session_id=42,
                verbose=False,
                html=False,
                log_experiment=False,
                system_log=False,
            )

            self.logger.info(f"[ModelTrainer] Comparing regression models (top {n_models})...")

            best_models = compare_models_reg(
                n_select=n_models,
                sort=metric or "R2",
                verbose=False,
            )

            comparison_df = pull_reg()

            if not isinstance(best_models, list):
                best_models = [best_models]

            best_model = best_models[0]
            final_model = finalize_model_reg(best_model)
            model_path = self.models_dir / "best_regression_model.pkl"
            save_model_reg(final_model, str(model_path.with_suffix("")))

            self.logger.info(f"[ModelTrainer] Best regression model saved to {model_path}")

            # Prepare results
            results = {
                "timestamp": datetime.now().isoformat(),
                "task_type": "regression",
                "target_column": target,
                "n_models_compared": len(best_models),
                "best_model": str(type(best_model).__name__),
                "optimize_metric": metric or "R2",
                "model_path": str(model_path),
                "comparison_results": comparison_df.to_dict("records"),
            }

            return results

        except Exception as e:
            self.logger.error(f"[ModelTrainer] Error in regression training: {e}", exc_info=True)
            return {"error": str(e)}
