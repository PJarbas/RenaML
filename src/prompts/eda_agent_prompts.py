"""Prompts for EDA Agent."""

AGENT_INSTRUCTIONS = [
    "You are an expert in data quality analysis and exploratory data analysis.",
    "Analyze data critically for quality issues, patterns, and anomalies.",
    "Identify outliers, distributions, and interesting patterns in data.",
    "Provide actionable insights for feature engineering and modeling.",
    "Suggest data cleaning strategies based on validation results.",
]

ANALYSIS_PROMPT_TEMPLATE = """
Analyze this dataset comprehensively:

**Data Quality:**
- Shape: {rows} rows, {columns} columns
- Memory Usage: {memory_mb:.2f} MB
- Duplicates: {duplicates}
- Null Percentage: {null_pct:.2f}%
- Columns with nulls: {cols_with_nulls}

**Data Structure:**
- Numeric columns ({num_numeric}): {numeric_cols}
- Categorical columns ({num_categorical}): {categorical_cols}

**Issues Found:**
{alerts}

**Missing Values:**
{missing_values}

Provide a comprehensive analysis including:
1. **Data Quality Assessment** (2-3 sentences)
2. **Key Insights** from distributions and patterns
3. **Top 5 Recommendations** for data preparation and feature engineering
4. **Potential Modeling Challenges** to address
5. **Suggested Next Steps** for the ML pipeline

Be specific and actionable.
"""
