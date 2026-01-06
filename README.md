# mech-util

Shared utilities and schemas for mechanical design agents.

## Purpose

This repository contains common code shared across all agent repositories:
- **Inter-repo schemas**: Communication contracts between agents
- **LLM client utilities**: vLLM server connection and management
- **Prompt manager**: Versioned prompt template loading and rendering
- **Local model server scripts**: PBS job scripts for running vLLM servers

## Installation

```bash
cd mech-util
pip install -e .
```

## Usage

### Schemas

```python
from mech_util.schemas import ComponentIntent, InterfaceContract

component = ComponentIntent(
    component_id="comp_bracket",
    parent_program_id="prog_test",
    component_name="Bracket",
    functional_description="Mounting bracket for motor"
)
```

### LLM Client

```python
from mech_util.tools import create_llm_client, get_llm_config

# Create LLM client (auto-discovers model from server)
llm = create_llm_client()

# Get config for passing to agents
config = get_llm_config()
```

### Prompt Manager

```python
from mech_util.tools import PromptManager

pm = PromptManager(prompts_dir="path/to/prompts")
rendered = pm.render_prompt("lme.clarify.v1", {"intent": "..."})
```

## Structure

```
mech-util/
├── schemas/          # Inter-agent communication schemas
│   ├── agent_component.py
│   ├── agent_assembly.py
│   ├── agent_simulation.py
│   └── interface_contract.py
├── tools/            # Shared utilities
│   ├── llm_client.py
│   ├── llm_config.py
│   └── prompt_manager.py
├── local_model_server/  # PBS scripts for vLLM servers
│   ├── mech_client.sh
│   ├── mech_client_setup.sh
│   ├── mech_server.pbs
│   ├── mech_server_llm.pbs
│   └── mech_server_vlm.pbs
└── pyproject.toml
```

## Dependencies

- `pydantic>=2.0`: Schema validation
- `langchain-core>=0.3.0`: LLM client base
- `langchain-openai>=0.2.0`: OpenAI-compatible API client
- `pyyaml>=6.0`: YAML parsing
- `jinja2>=3.1`: Template rendering
- `requests>=2.31`: HTTP client

