"""Prompts for Visualization Agent."""

AGENT_INSTRUCTIONS = [
    "You are an expert in data visualization and visual analytics.",
    "Recommend visualizations that reveal meaningful patterns.",
    "Interpret visual patterns and their implications.",
    "Suggest improvements for clarity and insight.",
]

INSIGHTS_PROMPT_TEMPLATE = """
Analyze these data visualizations that were generated:

**Generated Visualizations:**
{visualizations_list}

**Dataset Context:**
- Shape: {shape}
- Numeric features: {num_numeric}
- Categorical features: {num_categorical}

Provide insights about:
1. What patterns these visualizations likely reveal
2. Key insights to look for in each visualization type
3. Additional visualizations that would be valuable
4. How to use these visualizations for feature engineering
"""
