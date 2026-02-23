# System Architecture Structure

This repository now includes a scaffolded `src/`-layout package that mirrors the System Architecture v1.0.0 plan. The code is intentionally stubbed so you can fill in logic later while keeping clear module boundaries.

**Module Tree**
```
src/
  core/
    __init__.py
    main.py
    py.typed
    models/
      __init__.py
      state.py
      signals.py
      memory_records.py
    llm/
      __init__.py
      client.py
      types.py
      providers/
        __init__.py
        openai.py
        anthropic.py
        gemini.py
    memory/
      __init__.py
      manager.py
      working_memory.py
      self_state.py
      prediction_error.py
      policy_traces.py
    layers/
      __init__.py
      interoceptive.py
      predictive.py
      action_selection.py
      metacognitive.py
    coordination/
      __init__.py
      messages.py
      workspace.py
    adapters/
      __init__.py
      minedojo/
        __init__.py
        env_adapter.py
        observation_mapper.py
        action_mapper.py
    runtime/
      __init__.py
      loop.py
    observability/
      __init__.py
      config.py
      paths.py
      serializer.py
      logger.py
      exceptions.py
```

**Core Layers**
Layer 1: Interoceptive Foundation
Responsibilities: Vital state tracking, allostatic prediction, arousal/valence computation.
Implementation: `src/core/layers/interoceptive.py`

Layer 2: Predictive Processing Engine
Responsibilities: World model prediction, prediction error calculation, precision weighting.
Implementation: `src/core/layers/predictive.py`

Layer 3: Action Selection and Agency
Responsibilities: Policy proposals, free-energy minimization, motor translation.
Implementation: `src/core/layers/action_selection.py`

Layer 4: Metacognitive Monitor
Responsibilities: Information integration, uncertainty estimation, goal coherence.
Implementation: `src/core/layers/metacognitive.py`

**Layer Modules**
The layer modules map directly to architecture responsibilities without a duplicate
agent wrapper layer:
- `src/core/layers/interoceptive.py`
- `src/core/layers/predictive.py`
- `src/core/layers/action_selection.py`
- `src/core/layers/metacognitive.py`

**Memory Subsystems**
Working Memory Buffer: Short-term active goals and predictions.
Implementation: `src/core/memory/working_memory.py`

Self-State Memory: Records past self-state snapshots.
Implementation: `src/core/memory/self_state.py`

Prediction Error History: Records prediction failures and surprises.
Implementation: `src/core/memory/prediction_error.py`

Policy Traces: Records action-outcome patterns.
Implementation: `src/core/memory/policy_traces.py`

Memory Manager: Unified access to all memory subsystems.
Implementation: `src/core/memory/manager.py`

**Coordination and Workspace**
Message Types: Lightweight inter-agent messages.
Implementation: `src/core/coordination/messages.py`

Global Workspace: Central broadcast buffer for agent signals.
Implementation: `src/core/coordination/workspace.py`

**MineDojo Integration**
Environment Adapter: Thin abstraction over MineDojo env lifecycle.
Implementation: `src/core/adapters/minedojo/env_adapter.py`

Observation Mapper: Maps raw env outputs into `AgentState`.
Implementation: `src/core/adapters/minedojo/observation_mapper.py`

Action Mapper: Maps internal action proposals to env actions.
Implementation: `src/core/adapters/minedojo/action_mapper.py`

**LLM Interface (Provider-Agnostic)**
LLM types and protocol: Standard chat-style messages and request/response schema.
Implementation: `src/core/llm/types.py`, `src/core/llm/client.py`

Provider stubs: No SDK dependencies; placeholders for OpenAI/Anthropic/Gemini.
Implementation: `src/core/llm/providers/*.py`

**Runtime Loop**
AgentLoop: Handles environment reset/step, state extraction, and observability.
Implementation: `src/core/runtime/loop.py`

Entrypoint: `src/core/main.py` defines a stub `main()` for future wiring.

**Data Flow Example: Unexpected Mob**
1. Predictive layer detects prediction error: unexpected mob spawn.
2. Metacognitive layer broadcasts a threat signal to the workspace.
3. Interoceptive layer raises arousal and generates survival goals.
4. Action-selection layer proposes actions (fight, flee, build barrier).
5. Memory Manager records prediction errors and policy traces.

**Observability and Logs**
Run logger: Creates a per-execution folder and writes JSONL logs for events, LLM requests/responses, memory activity, and state snapshots, plus tracebacks.
Implementation: `src/core/observability/logger.py`

Exception hooks: Captures uncaught exceptions and fatal errors.
Implementation: `src/core/observability/exceptions.py`

Run folder layout (default `logs/runs/`):
- `run.json`: Run metadata
- `events.jsonl`: Lifecycle and generic events
- `llm.jsonl`: Provider-agnostic LLM request/response logs
- `memory.jsonl`: Memory write/query events
- `state.jsonl`: AgentState snapshots
- `tracebacks.log`: Human-readable tracebacks
- `tracebacks.jsonl`: Structured exception events

LLM log schema (example fields):
- `provider`, `model`, `messages`, `request_params` on request
- `provider`, `model`, `text`, `usage`, `latency_ms`, `raw` on response
Note: Hidden chain-of-thought is never captured; only explicit fields are logged.

**How to Run (Future)**
Use a `src/`-aware invocation once you implement the runtime logic.
```
PYTHONPATH=src python -m core.main
```
