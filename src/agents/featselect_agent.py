"""FeatSelectAgent: Performs feature selection using multiple methods with LLM analysis."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from prompts.featselect_agent_prompts import AGENT_INSTRUCTIONS, RECOMMENDATIONS_PROMPT_TEMPLATE


class FeatSelectAgent:
    """Agent responsible for feature selection with LLM-powered recommendations."""

    def __init__(self, team_memory: dict[str, Any], run_dir: Path) -> None:
        self.team_memory = team_memory
        self.run_dir = run_dir
        self.logger = logging.getLogger(__name__)
        self.feat_dir = run_dir / "feature_selection"
        self.feat_dir.mkdir(parents=True, exist_ok=True)

        # Create Agent instance for LLM recommendations
        self.llm_agent = Agent(
            name="FeatureSelector",
            role="feature_selector",
            model=OpenAIChat(id="gpt-4o"),
            instructions=AGENT_INSTRUCTIONS,
        )

    def run(
        self, target_column: str, task_type: str = "classification", top_k: int = 20
    ) -> dict[str, Any]:
        self.logger.info(f"[FeatureSelector] Starting feature selection for {task_type} task")

        try:
            # Get dataset from team memory
            dataset_path = self.team_memory.get("dataset_path")
            if not dataset_path:
                raise ValueError("No dataset_path found in team memory")

            df = pd.read_parquet(dataset_path)
            self.logger.info(f"[FeatureSelector] Loaded dataset with shape {df.shape}")

            # Prepare features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")

            X, y = self._prepare_data(df, target_column)

            # Run multiple feature selection methods
            results = {}

            # 1. Statistical methods (filter-based)
            results["statistical"] = self._statistical_selection(X, y, task_type, top_k)

            # 2. Tree-based importance (embedded)
            results["tree_based"] = self._tree_based_selection(X, y, task_type, top_k)

            # 3. Mutual information
            results["mutual_info"] = self._mutual_info_selection(X, y, task_type, top_k)

            # 4. Combine methods and rank features
            combined_ranking = self._combine_rankings(results, top_k)

            # 5. Get LLM recommendations
            llm_recommendations = self._get_llm_recommendations(
                results, combined_ranking, target_column, task_type, top_k
            )

            # Save results
            report_path = self.feat_dir / "feature_selection_report.json"
            report = {
                "timestamp": datetime.now().isoformat(),
                "target_column": target_column,
                "task_type": task_type,
                "total_features": len(X.columns),
                "top_k": top_k,
                "methods": results,
                "combined_ranking": combined_ranking,
                "llm_recommendations": llm_recommendations,
            }

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            # Update team memory
            feature_list = [feat for feat, _ in combined_ranking[:top_k]]
            self.team_memory["feature_list"] = feature_list
            self.team_memory["feature_ranking"] = combined_ranking
            self.team_memory["target_column"] = target_column
            self.team_memory["task_type"] = task_type
            self.team_memory["feature_selection_insights"] = llm_recommendations

            self.logger.info(f"[FeatureSelector] Selected {len(feature_list)} features")

            return {
                "status": "success",
                "selected_features": feature_list,
                "ranking": combined_ranking,
                "report_path": str(report_path),
                "llm_recommendations": llm_recommendations,
            }

        except Exception as e:
            self.logger.error(
                f"[FeatureSelector] Error during feature selection: {e}", exc_info=True
            )
            return {
                "status": "error",
                "error": str(e),
            }

    def _prepare_data(self, df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
        # Separate target
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Handle numeric features only for now (simple approach)
        numeric_cols = X.select_dtypes(include=["number"]).columns
        X_numeric = X[numeric_cols].copy()

        # Fill missing values with median
        X_numeric = X_numeric.fillna(X_numeric.median())

        # Handle infinite values
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
        X_numeric = X_numeric.fillna(X_numeric.median())

        self.logger.info(f"[FeatureSelector] Using {len(X_numeric.columns)} numeric features")

        return X_numeric, y

    def _get_llm_recommendations(
        self,
        results: dict[str, dict[str, float]],
        combined_ranking: list[tuple],
        target_column: str,
        task_type: str,
        top_k: int,
    ) -> str:
        try:
            # Prepare summary of results
            top_features = [
                f"{feat} (avg_rank: {rank:.2f})" for feat, rank in combined_ranking[:top_k]
            ]

            method_agreements = []
            for _i, (feat, _) in enumerate(combined_ranking[:10]):
                method_ranks = []
                for method_name, method_results in results.items():
                    if feat in method_results:
                        rank = list(method_results.keys()).index(feat) + 1
                        method_ranks.append(f"{method_name}: #{rank}")
                method_agreements.append(f"{feat}: {', '.join(method_ranks)}")

            prompt = RECOMMENDATIONS_PROMPT_TEMPLATE.format(
                task_type=task_type,
                target_column=target_column,
                top_k=top_k,
                top_features=chr(10).join(top_features),
                method_agreements=chr(10).join(method_agreements),
                total_features=len(combined_ranking),
            )

            self.logger.info(
                "[FeatureSelector] Requesting LLM analysis of feature selection results"
            )
            response = self.llm_agent.run(input=prompt)
            llm_text = response.content

            self.logger.info("[FeatureSelector] LLM recommendations generated")
            return llm_text

        except Exception as e:
            self.logger.error(f"[FeatureSelector] Error getting LLM recommendations: {e}")
            return f"Error generating LLM recommendations: {str(e)}"

    def _statistical_selection(
        self, X: pd.DataFrame, y: pd.Series, task_type: str, top_k: int
    ) -> dict[str, float]:
        try:
            if task_type == "classification":
                selector = SelectKBest(score_func=f_classif, k=min(top_k, len(X.columns)))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(top_k, len(X.columns)))

            selector.fit(X, y)

            scores = dict(zip(X.columns, selector.scores_, strict=True))
            sorted_scores = {
                k: float(v) for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            }

            self.logger.info("[FeatureSelector] Statistical selection complete")
            return sorted_scores

        except Exception as e:
            self.logger.error(f"[FeatureSelector] Error in statistical selection: {e}")
            return {}

    def _tree_based_selection(
        self, X: pd.DataFrame, y: pd.Series, task_type: str, top_k: int
    ) -> dict[str, float]:
        try:
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

            model.fit(X, y)

            importances = dict(zip(X.columns, model.feature_importances_, strict=True))
            sorted_importances = {
                k: float(v)
                for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)
            }

            self.logger.info("[FeatureSelector] Tree-based selection complete")
            return sorted_importances

        except Exception as e:
            self.logger.error(f"[FeatureSelector] Error in tree-based selection: {e}")
            return {}

    def _mutual_info_selection(
        self, X: pd.DataFrame, y: pd.Series, task_type: str, top_k: int
    ) -> dict[str, float]:
        try:
            if task_type == "classification":
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)

            scores = dict(zip(X.columns, mi_scores, strict=True))
            sorted_scores = {
                k: float(v) for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            }

            self.logger.info("[FeatureSelector] Mutual information selection complete")
            return sorted_scores

        except Exception as e:
            self.logger.error(f"[FeatureSelector] Error in mutual info selection: {e}")
            return {}

    def _combine_rankings(self, results: dict[str, dict[str, float]], top_k: int) -> list[tuple]:
        # Collect all features
        all_features = set()
        for method_results in results.values():
            all_features.update(method_results.keys())

        # Calculate average rank for each feature
        feature_ranks = {}
        for feature in all_features:
            ranks = []
            for _method_name, method_results in results.items():
                if feature in method_results:
                    # Get rank (1-indexed)
                    rank = list(method_results.keys()).index(feature) + 1
                    ranks.append(rank)
                else:
                    # If feature not in method, give it worst rank
                    ranks.append(len(all_features))

            # Average rank
            feature_ranks[feature] = np.mean(ranks)

        # Sort by average rank (lower is better)
        sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1])

        self.logger.info(f"[FeatureSelector] Combined rankings from {len(results)} methods")

        return sorted_features
