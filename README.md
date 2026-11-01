# LangChain Tutorial

## Setup

From the project root, create the `.venv` environment and install dependencies:

```sh
uv venv
uv sync
```

Create your local `.env` file from the template, then add the API key for the provider selected by `USE_LLM`.

**Windows PowerShell**

```powershell
Copy-Item .env.example .env
```

**Linux**

```sh
cp .env.example .env
```

The `.env` file is git-ignored; keep API keys there and do not commit them.

Activate the environment if you want to run `python` directly.

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux**

```sh
source .venv/bin/activate
```

You can also run commands without activating the environment:

```sh
uv run python src/00-hello-world.py
```

## Shared LLM client

Set `USE_LLM` in `.env` to `groq`, `openai`, or `gemini`, and provide the matching API key. Model names can be changed with `GROQ_MODEL`, `OPENAI_MODEL`, or `GEMINI_MODEL`.

Import the configured chat model in a script under `src/`:

```python
from global_llm_client import llm
```

Run the examples with `uv run python src/00-hello-world.py`.
