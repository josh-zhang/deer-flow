# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state (read first)

- **Backend (`src/`) is work-in-progress.** Expect bugs, unfinished code paths, and references that don't resolve. For example, `src/graph/builder.py` currently imports node functions from `src.graph.cp_nodes`, but the implementations live in `src/graph/nodes.py` — that import is part of an in-flight refactor, not a typo to fix. There are also untracked WIP files (`src/graph/xb_node.py`, `src/prompts/code{2..8}/`, etc.). **Do not "clean up" broken-looking backend code without asking the user first** — it is likely intentional WIP.
- **Frontend (`web/`) is complete but outdated.** It is internally consistent but has not been re-synced to recent backend changes, so API contracts between `web/` and `src/server/` may drift. Suspect the sync gap before assuming a bug.

## Commands

### Backend (Python 3.12+, managed by `uv`)
- `uv sync` — install deps
- `make install-dev` — add dev + test extras (`ruff`, `pytest`, `langgraph-cli`)
- `make serve` — run FastAPI server with reload (`uv run server.py --reload`, binds `localhost:8000`)
- `uv run main.py` — interactive CLI (built-in questions); `uv run main.py "<query>"` for one-shot
- `make test` / `make coverage` — pytest under `tests/` (Note: the `tests/` directory does not currently exist in the repo; `test_fix.py` at root is the only test file. Running `make test` will fail until a `tests/` directory is created.)
- `uv run pytest path/to/test_file.py::test_name` — single test
- `make format` / `make lint` — ruff format / `ruff check --fix --select I`
- `make langgraph-dev` — LangGraph Studio against the graphs declared in `langgraph.json`
- Coverage gate: `fail_under = 25` in `pyproject.toml`

### Frontend (`web/`, Node 22+, pnpm)
- `cd web && pnpm dev` — Next.js dev server (consumes `../.env` via dotenv-cli)
- `cd web && pnpm test:run` (non-watch) / `pnpm test` (watch) — Jest
- `make lint-frontend` — runs `pnpm lint && pnpm typecheck && pnpm test:run && pnpm build`
- `./bootstrap.sh -d` (macOS/Linux) — backend + web together in dev mode

### Configuration files (both required before running)
- `cp .env.example .env` — secrets and feature flags (search/RAG provider keys, `SEARCH_API`, `LANGGRAPH_CHECKPOINT_DB_URL`, `AGENT_RECURSION_LIMIT`, `DEBUG`)
- `cp conf.yaml.example conf.yaml` — LLM provider config keyed by tier (`BASIC_MODEL`, `REASONING_MODEL`, `VISION_MODEL`, `CODE_MODEL`). Changing `conf.yaml` requires a server restart.

## Architecture

### Top-level entry points
- `main.py` — CLI; calls `run_agent_workflow_async` in `src/workflow.py`
- `server.py` → `src.server.app:app` — FastAPI app; constructs a graph with Mongo/Postgres checkpointing and exposes `/api/chat/stream` plus podcast/PPT/prose/prompt-enhancer/TTS/RAG/MCP endpoints
- `langgraph.json` declares three graphs for LangGraph Studio: `deep_research` (main), `podcast_generation`, `ppt_generation`

### Main research graph (`src/graph/builder.py`)
A single StateGraph supports two pipeline modes selected by `state["pipeline_mode"]`:
- `"bi"` (business investigation, default) — goes through `rule_splitter` and `arbitrator` before `analyst`
- `"cp"` (compliance pipeline) — skips both; `curator → analyst` directly, and `plan_validator` can route straight to `analyst`

Node sequence (all nodes registered; routing functions decide which fire per mode):
```
START → coordinator → [background_investigator] → planner ↔ human_feedback
       → plan_validator → researcher → curator
       → (BI) rule_splitter → arbitrator → analyst
       → (CP)                              analyst
       → analyst → reporter → END  (or → planner if analyst sets replanning_needed)
```
The four conditional routers (`route_from_validator`, `route_from_curator`, `route_from_splitter`, `route_from_analyst`) all branch on `pipeline_mode` and on which plan steps still have `execution_res is None`. When extending the graph, update both the node imports in `builder.py` and the corresponding `route_from_*` function — many edges are conditional, not fixed.

Node implementations live in `src/graph/nodes.py`. Plan/step data classes are in `src/models/base.py` (`Plan`, `Resource`); structured planner output schemas are in `src/graph/planner_model.py`; curator output schemas are in `src/graph/curator_models.py` / `curator_parser.py`.

### State (`src/graph/types.py`)
`State` extends LangGraph's `MessagesState`. Notable cross-node accumulators:
- `current_plan: Plan` — the active plan; steps carry `step_type` ∈ {`research`, `analysis`} and `execution_res`
- `searcher_results` / `searcher_summaries` — researcher output appended across steps
- `document_chunk_maps` / `document_metadata` — chunked source documents shared from curator → rule_splitter / analyst for citation lookup
- `citations` — appended globally with `operator.add`
- `replan_iterations` / `max_replan_iterations` — bounds the analyst→planner loop (BI default 3, CP default 2)
- `auto_accepted_plan`, `goto` — UI/human-in-the-loop control

### Agent factory (`src/agents/agents.py`)
`create_agent()` wraps `langchain.agents.create_agent` with two middlewares:
- `DynamicPromptMiddleware` — renders the named Jinja2 prompt template before each model call (locale captured in closure, not state — per issue #743)
- `PreModelHookMiddleware` — optional legacy hook, supports sync or async (sync runs on `asyncio.to_thread`)

`AGENT_LLM_MAP` in `src/config/agents.py` maps each agent name to an LLM tier (`basic` / `reasoning` / `vision` / `code`); tiers are resolved through `get_llm_by_type` against `conf.yaml`. Tools requiring user approval before execution can be listed in `interrupt_before_tools` and are wrapped via `wrap_tools_with_interceptor`.

### Prompt system (`src/prompts/template.py`)
Templates are Jinja2 markdown files in `src/prompts/`. Resolution **always tries `<name>.zh_CN.md` first**, then falls back to `<name>.md` — meaning a Chinese-locale prompt will be used even when `locale="en-US"` if both exist. State and `Configuration` fields are passed as template variables, plus `CURRENT_TIME`. Every rendered system prompt has a trailing instruction enforcing Unicode math symbols.

### Tools (`src/tools/`)
Search engines selected via `SEARCH_API` env var (`tavily` / `infoquest` / `duckduckgo` / `brave_search` / `arxiv` / `searx`). Crawler engine selected in `conf.yaml` (`jina` default, `infoquest`). `python_repl.py` is the Coder tool. MCP servers can be declared in `RunnableConfig["configurable"]["mcp_settings"]` (see `src/workflow.py` for the in-code example with `mcp-github-trending`) and are loaded by `src/server/mcp_utils.load_mcp_tools`.

### RAG (`src/rag/`)
Provider selected by `RAG_PROVIDER` env var; supported: `ragflow`, `qdrant`, `milvus`, `vikingdb`, `dify`, `moi`, `mskb`. `Resource` objects flow through `State.resources` and are passed to retriever calls.

### Auxiliary sub-graphs
Each is its own LangGraph in `src/<feature>/graph/builder.py`:
- `src/podcast/` — script writer + TTS pipeline
- `src/ppt/` — Marp-based slide generation (requires `marp-cli` installed system-wide)
- `src/prose/` — Notion-style block edits (polish / shorten / expand)
- `src/prompt_enhancer/` — refines user prompts before planner

### Persistence
`src.server.app` chooses a checkpointer at startup based on env: `AsyncPostgresSaver` (`LANGGRAPH_CHECKPOINT_DB_URL` starts with `postgres://`), `AsyncMongoDBSaver` (Mongo URI), or in-memory `MemorySaver`. `src/graph/checkpoint.py` (`ChatStreamManager`) layers a separate chat-stream message store on top, also dual-backed.

## Repo conventions worth knowing

- Pre-commit hook in `pre-commit` (not auto-installed): `chmod +x pre-commit && ln -s ../../pre-commit .git/hooks/pre-commit`
- README is also translated to zh / ja / de / es / pt / ru — keep wording in sync when editing the English README only if the change is substantive.
- `docs/configuration_guide.md`, `docs/API.md`, `docs/mcp_integrations.md` are the canonical references for those areas.
- `Agent.md` and `.github/copilot-instructions.md` carry overlapping but slightly more general guidance for other AI assistants; prefer this file over those if they conflict for Claude Code specifically.
