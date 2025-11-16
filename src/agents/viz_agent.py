"""VizAgent: Creates interactive Plotly visualizations."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from plotly.subplots import make_subplots

from prompts.viz_agent_prompts import AGENT_INSTRUCTIONS, INSIGHTS_PROMPT_TEMPLATE


class VizAgent:
    """Agent responsible for creating visualizations with LLM insights."""

    def __init__(self, team_memory: dict[str, Any], run_dir: Path) -> None:
        self.team_memory = team_memory
        self.run_dir = run_dir
        self.logger = logging.getLogger(__name__)
        self.viz_dir = run_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Create Agent instance for LLM insights
        self.llm_agent = Agent(
            name="VizMaker",
            role="Data visualization and insight specialist",
            model=OpenAIChat(id="gpt-4o"),
            instructions=AGENT_INSTRUCTIONS,
            markdown=True,
        )

    def run(self) -> dict[str, Any]:
        self.logger.info("[VizMaker] Starting visualization generation")

        try:
            # Get dataset from team memory
            dataset_path = self.team_memory.get("dataset_path")
            if not dataset_path:
                raise ValueError("No dataset_path found in team memory")

            df = pd.read_parquet(dataset_path)
            self.logger.info(f"[VizMaker] Loaded dataset with shape {df.shape}")

            # Get EDA summary for context
            eda_summary = self.team_memory.get("eda_summary", {})

            # Generate visualizations
            plots_index = []

            # 1. Missing values heatmap
            if "missing_values" in eda_summary and eda_summary["missing_values"]:
                plot_path = self._create_missing_values_plot(df)
                if plot_path:
                    plots_index.append(
                        {
                            "type": "missing_values",
                            "path": str(plot_path),
                            "description": "Missing values heatmap",
                        }
                    )

            # 2. Distribution plots for numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                plot_path = self._create_distributions_plot(df, numeric_cols)
                if plot_path:
                    plots_index.append(
                        {
                            "type": "distributions",
                            "path": str(plot_path),
                            "description": "Numeric feature distributions",
                        }
                    )

            # 3. Correlation matrix
            if len(numeric_cols) > 1:
                plot_path = self._create_correlation_matrix(df, numeric_cols)
                if plot_path:
                    plots_index.append(
                        {
                            "type": "correlation",
                            "path": str(plot_path),
                            "description": "Correlation matrix",
                        }
                    )

            # 4. Categorical value counts
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if categorical_cols:
                plot_path = self._create_categorical_plots(df, categorical_cols)
                if plot_path:
                    plots_index.append(
                        {
                            "type": "categorical",
                            "path": str(plot_path),
                            "description": "Categorical feature distributions",
                        }
                    )

            # Save plots index
            index_path = self.viz_dir / "plots_index.json"
            with open(index_path, "w") as f:
                json.dump(plots_index, f, indent=2)

            # Get LLM interpretation of visualizations
            llm_insights = self._get_llm_insights(plots_index, df)

            # Update team memory
            self.team_memory["plots_index"] = plots_index
            self.team_memory["viz_dir"] = str(self.viz_dir)
            self.team_memory["viz_llm_insights"] = llm_insights

            self.logger.info(f"[VizMaker] Generated {len(plots_index)} visualizations")

            return {
                "status": "success",
                "plots_count": len(plots_index),
                "plots_index": plots_index,
                "llm_insights": llm_insights,
            }

        except Exception as e:
            self.logger.error(f"[VizMaker] Error during visualization: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def _create_missing_values_plot(self, df: pd.DataFrame) -> Path | None:
        try:
            self.logger.info("[VizMaker] Creating missing values plot...")

            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)

            # Filter to show only columns with missing values
            missing_data = pd.DataFrame(
                {
                    "Column": missing[missing > 0].index,
                    "Missing Count": missing[missing > 0].values,
                    "Missing %": missing_pct[missing > 0].values,
                }
            ).sort_values("Missing %", ascending=False)

            fig = px.bar(
                missing_data,
                x="Column",
                y="Missing %",
                title="Missing Values by Column",
                labels={"Missing %": "Missing Percentage"},
                text="Missing %",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=500)

            output_path = self.viz_dir / "missing_values.html"
            fig.write_html(output_path, include_plotlyjs='cdn')

            return output_path

        except Exception as e:
            self.logger.error(f"[VizMaker] Error creating missing values plot: {e}")
            return None

    def _create_distributions_plot(self, df: pd.DataFrame, numeric_cols: list[str]) -> Path | None:
        try:
            self.logger.info("[VizMaker] Creating distribution plots...")

            # Limit to first 10 numeric columns for readability
            cols_to_plot = numeric_cols[:10]

            rows = (len(cols_to_plot) + 2) // 3
            fig = make_subplots(rows=rows, cols=3, subplot_titles=cols_to_plot)

            for idx, col in enumerate(cols_to_plot):
                row = idx // 3 + 1
                col_idx = idx % 3 + 1

                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row,
                    col=col_idx,
                )

            fig.update_layout(
                height=300 * rows,
                title_text="Numeric Feature Distributions",
                showlegend=False,
            )

            output_path = self.viz_dir / "distributions.html"
            fig.write_html(output_path, include_plotlyjs='cdn')

            return output_path

        except Exception as e:
            self.logger.error(f"[VizMaker] Error creating distribution plots: {e}")
            return None

    def _create_correlation_matrix(self, df: pd.DataFrame, numeric_cols: list[str]) -> Path | None:
        try:
            self.logger.info("[VizMaker] Creating correlation matrix...")

            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix",
            )
            fig.update_layout(height=800, width=900)

            output_path = self.viz_dir / "correlation_matrix.html"
            fig.write_html(output_path, include_plotlyjs='cdn')

            return output_path

        except Exception as e:
            self.logger.error(f"[VizMaker] Error creating correlation matrix: {e}")
            return None

    def _create_categorical_plots(
        self, df: pd.DataFrame, categorical_cols: list[str]
    ) -> Path | None:
        try:
            self.logger.info("[VizMaker] Creating categorical plots...")

            # Limit to first 6 categorical columns and top 10 values each
            cols_to_plot = categorical_cols[:6]

            rows = (len(cols_to_plot) + 1) // 2
            fig = make_subplots(rows=rows, cols=2, subplot_titles=cols_to_plot)

            for idx, col in enumerate(cols_to_plot):
                row = idx // 2 + 1
                col_idx = idx % 2 + 1

                value_counts = df[col].value_counts().head(10)

                fig.add_trace(
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        name=col,
                        showlegend=False,
                    ),
                    row=row,
                    col=col_idx,
                )

            fig.update_layout(
                height=400 * rows,
                title_text="Categorical Feature Distributions (Top 10 values)",
                showlegend=False,
            )

            output_path = self.viz_dir / "categorical_distributions.html"
            fig.write_html(output_path, include_plotlyjs='cdn')

            return output_path

        except Exception as e:
            self.logger.error(f"[VizMaker] Error creating categorical plots: {e}")
            return None

    def _get_llm_insights(self, plots_index: list[dict], df: pd.DataFrame) -> str:
        try:
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            visualizations_list = chr(10).join([f"- {p['type']}: {p['description']}" for p in plots_index])

            prompt = INSIGHTS_PROMPT_TEMPLATE.format(
                visualizations_list=visualizations_list,
                shape=df.shape,
                num_numeric=len(numeric_cols),
                num_categorical=len(categorical_cols),
            )

            response = self.llm_agent.run(input=prompt)
            return response.content

        except Exception as e:
            self.logger.warning(f"[VizMaker] LLM insights failed: {e}")
            return "LLM insights not available"
