"""EDAAgent: Handles data ingestion and generates comprehensive EDA reports."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sweetviz as sv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from ydata_profiling import ProfileReport

from prompts.eda_agent_prompts import AGENT_INSTRUCTIONS, ANALYSIS_PROMPT_TEMPLATE


class EDAAgent:
    """Agent responsible for data ingestion and exploratory data analysis."""

    def __init__(self, team_memory: dict[str, Any], run_dir: Path) -> None:
        self.team_memory = team_memory
        self.run_dir = run_dir
        self.logger = logging.getLogger(__name__)
        self.eda_dir = run_dir / "eda"
        self.eda_dir.mkdir(parents=True, exist_ok=True)

        # Create Agent instance for LLM analysis
        self.llm_agent = Agent(
            name="EDAExplorer",
            role="Data ingestion and exploratory data analysis specialist",
            model=OpenAIChat(id="gpt-4o"),
            instructions=AGENT_INSTRUCTIONS,
            markdown=True,
        )

    def run(self, csv_path: str) -> dict[str, Any]:
        self.logger.info("[EDAExplorer] Starting data ingestion and EDA analysis")

        try:
            # 1. Ingest data
            df = pd.read_csv(csv_path)
            self.logger.info(f"[EDAExplorer] Loaded CSV with shape {df.shape}")

            # 2. Validate and save data
            validation_results = self._validate_data(df)

            # Convert to Parquet for better performance
            parquet_path = self.run_dir / "data_clean.parquet"
            df.to_parquet(parquet_path, index=False)
            self.logger.info(f"[EDAExplorer] Saved Parquet to {parquet_path}")

            # Generate schema
            schema = self._generate_schema(df)

            # Store in team memory
            self.team_memory["dataset_path"] = str(parquet_path)
            self.team_memory["schema"] = schema
            self.team_memory["raw_shape"] = df.shape

            # 3. Generate ydata-profiling report
            ydata_path = self._generate_ydata_profile(df)

            # 4. Generate SweetViz report
            sweetviz_path = self._generate_sweetviz_report(df)

            # 5. Generate custom summary
            summary = self._generate_summary(df)

            # 6. Get LLM analysis combining ingestion + EDA insights
            llm_analysis = self._get_llm_analysis(df, validation_results, summary)
            summary["llm_analysis"] = llm_analysis

            # Update team memory
            self.team_memory["eda.ydata_html"] = str(ydata_path) if ydata_path else None
            self.team_memory["eda.sweetviz_html"] = str(sweetviz_path) if sweetviz_path else None
            self.team_memory["eda_summary"] = summary
            self.team_memory["ingest_summary"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "validation": validation_results,
                "ingestion_timestamp": datetime.now().isoformat(),
            }

            self.logger.info("[EDAExplorer] Data ingestion and EDA complete")

            return {
                "status": "success",
                "dataset_path": str(parquet_path),
                "ydata_report": str(ydata_path) if ydata_path else None,
                "sweetviz_report": str(sweetviz_path) if sweetviz_path else None,
                "summary": summary,
                "validation": validation_results,
                "llm_analysis": llm_analysis,
            }

        except Exception as e:
            self.logger.error(f"[EDAExplorer] Error during ingestion/EDA: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def _validate_data(self, df: pd.DataFrame) -> dict[str, Any]:
        validation = {
            "has_duplicates": bool(df.duplicated().any()),
            "duplicate_count": int(df.duplicated().sum()),
            "columns_with_nulls": int(df.isnull().any().sum()),
            "total_nulls": int(df.isnull().sum().sum()),
            "null_percentage": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
        }

        # Check for columns with high null percentage
        null_cols = df.isnull().sum()
        high_null_cols = null_cols[null_cols / len(df) > 0.5].to_dict()
        if high_null_cols:
            validation["high_null_columns"] = {k: int(v) for k, v in high_null_cols.items()}
            self.logger.warning(
                f"[EDAExplorer] Columns with >50% nulls: {list(high_null_cols.keys())}"
            )

        return validation

    def _generate_schema(self, df: pd.DataFrame) -> dict[str, str]:
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()

            schema[col] = {
                "dtype": dtype,
                "null_count": int(null_count),
                "null_percentage": float(null_count / len(df) * 100),
                "unique_count": int(unique_count),
            }

        return schema

    def _generate_ydata_profile(self, df: pd.DataFrame) -> Path | None:
        try:
            self.logger.info("[EDAExplorer] Generating ydata-profiling report...")

            profile = ProfileReport(
                df,
                title="RenaML EDA Report",
                explorative=True,
                minimal=False,
            )

            output_path = self.eda_dir / "ydata_profile.html"
            profile.to_file(output_path)

            self.logger.info(f"[EDAExplorer] ydata-profiling report saved to {output_path}")
            return output_path

        except ImportError:
            self.logger.warning("[EDAExplorer] ydata-profiling not available, skipping")
            return None
        except Exception as e:
            self.logger.error(f"[EDAExplorer] Error generating ydata profile: {e}")
            return None

    def _generate_sweetviz_report(self, df: pd.DataFrame) -> Path | None:
        try:
            self.logger.info("[EDAExplorer] Generating SweetViz report...")

            report = sv.analyze(df)
            output_path = self.eda_dir / "sweetviz_report.html"
            report.show_html(str(output_path), open_browser=False)

            self.logger.info(f"[EDAExplorer] SweetViz report saved to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"[EDAExplorer] Error generating SweetViz report: {e}")
            return None

    def _generate_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            summary["numeric_columns"] = numeric_cols
            summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            summary["categorical_columns"] = categorical_cols
            summary["categorical_unique_counts"] = {
                col: int(df[col].nunique()) for col in categorical_cols
            }

        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        summary["missing_values"] = {k: int(v) for k, v in missing[missing > 0].items()}
        summary["missing_percentages"] = {
            k: float(v) for k, v in missing_pct[missing_pct > 0].items()
        }

        # Alerts
        alerts = []
        high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
        if high_missing_cols:
            alerts.append(f"Columns with >50% missing: {high_missing_cols}")

        high_cardinality_cols = [
            col for col in categorical_cols if df[col].nunique() > len(df) * 0.5
        ]
        if high_cardinality_cols:
            alerts.append(f"High cardinality categorical columns: {high_cardinality_cols}")

        summary["alerts"] = alerts

        return summary

    def _get_llm_analysis(
        self, df: pd.DataFrame, validation: dict[str, Any], summary: dict[str, Any]
    ) -> str:
        try:
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(
                rows=df.shape[0],
                columns=df.shape[1],
                memory_mb=df.memory_usage(deep=True).sum() / 1024 / 1024,
                duplicates=validation["duplicate_count"],
                null_pct=validation["null_percentage"],
                cols_with_nulls=validation["columns_with_nulls"],
                num_numeric=len(summary.get("numeric_columns", [])),
                numeric_cols=", ".join(summary.get("numeric_columns", [])[:5]),
                num_categorical=len(summary.get("categorical_columns", [])),
                categorical_cols=", ".join(summary.get("categorical_columns", [])[:5]),
                alerts=chr(10).join(["- " + alert for alert in summary.get("alerts", [])]),
                missing_values=summary.get("missing_percentages", {}),
            )

            response = self.llm_agent.run(input=prompt)
            return response.content

        except Exception as e:
            self.logger.warning(f"[EDAExplorer] LLM analysis failed: {e}")
            return "LLM analysis not available"
