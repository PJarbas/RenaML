# Agent Prompts

This directory contains all the prompts used by the RenaML agents. Organizing prompts in separate files makes them easier to maintain, version, and modify.

## Structure

Each agent has its own prompt file:

- `eda_agent_prompts.py` - Prompts for the EDA Agent
- `featselect_agent_prompts.py` - Prompts for the Feature Selection Agent
- `modeling_agent_prompts.py` - Prompts for the Modeling Agent
- `viz_agent_prompts.py` - Prompts for the Visualization Agent
- `report_agent_prompts.py` - Prompts for the Report Agent

## Usage

Agents import prompts from these files instead of having hardcoded strings:

```python
from prompts.eda_agent_prompts import AGENT_INSTRUCTIONS, ANALYSIS_PROMPT_TEMPLATE

# Use in agent initialization
self.llm_agent = Agent(
    name="EDAExplorer",
    model=OpenAIChat(id="gpt-4o"),
    instructions=AGENT_INSTRUCTIONS,
)

# Use templates with formatting
prompt = ANALYSIS_PROMPT_TEMPLATE.format(
    rows=df.shape[0],
    columns=df.shape[1],
    # ... other parameters
)
```

## Benefits

1. **Maintainability** - All prompts in one place, easy to find and update
2. **Version Control** - Track changes to prompts separately from code logic
3. **Reusability** - Prompts can be shared across different parts of the codebase
4. **Testing** - Easier to test different prompt variations
5. **Collaboration** - Prompt engineers can work independently from developers

## Modifying Prompts

When modifying prompts:

1. Keep the format placeholders (e.g., `{rows}`, `{columns}`) consistent
2. Update the docstring if the prompt's purpose changes
3. Test the changes with representative data
4. Consider backward compatibility if other code depends on the prompt structure
