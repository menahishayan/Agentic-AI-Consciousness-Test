# Brain-Inspired Multi-Agent System — Animal AI Testbed

A game-agnostic brain architecture grounded in Anil Seth's *Beast Machine* / predictive processing theory, designed to survive in the Animal AI testbed and adaptable to any environment via a swap of one config value.

---

## Table of Contents

- [Theoretical Background](#theoretical-background)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Building Animal AI](#building-animal-ai)
- [Connecting to Animal AI](#connecting-to-animal-ai)
- [Configuration](#configuration)
- [Running](#running)
- [Observability](#observability)
- [Swapping Environments](#swapping-environments)
- [LLM Provider](#llm-provider)
- [Testing](#testing)

---

## Theoretical Background

The brain is modelled after predictive processing and allostatic regulation:

- **Interoceptive layer** — monitors internal physiological drives (health, saturation, energy, safety) and computes urgency via the *allostatic controller*
- **Predictive layer** — maintains an action-conditional world model (EMA transition table) and computes precision-weighted *prediction errors* per drive channel
- **Action selection layer** — a *free-energy minimiser* scores policies; an LLM is called selectively (5 trigger conditions) rather than on every step
- **Metacognitive monitor** — integrates signals from all layers, detects uncertainty spikes, broadcasts urgency
- **FAISS memory** — prediction error history, self-state tracking, and policy traces are stored in vector DBs; long-term policy outcomes persist as JSON across restarts

All layers communicate exclusively through a `GlobalWorkspace` message bus. No layer imports any concrete adapter.

---

## Architecture Overview

```
Environment (Animal AI / Headless / IoT)
        │
        ▼
AbstractEnvironmentAdapter  ←── only interface the brain touches
        │
        ├─ HomeostaticWrapper        (reward → health/saturation/energy deltas)
        ├─ ObservationMapper         (84×84 RGB → 11-dim visual feature vector)
        └─ ActionMapper              (policy_id → [movement, rotation] action tuple)
        │
        ▼
AgentLoop (runtime/loop.py)
        │
        ├─ Layer 1 · Interoceptive
        │   ├─ VitalStateMonitor
        │   ├─ AllostaticController   (urgency per drive channel)
        │   └─ ArousalValenceSystem
        │
        ├─ Layer 2 · Predictive
        │   ├─ WorldModelGenerator    (action-conditional EMA)
        │   └─ PredictionErrorCalculator
        │
        ├─ Layer 3 · Action Selection
        │   ├─ PolicyGenerator        (selective LLM + urgency fallback)
        │   ├─ FreeEnergyMinimizer
        │   └─ MotorControlInterface
        │
        ├─ Layer 4 · Metacognitive
        │   ├─ MetacognitiveMonitor
        │   └─ GoalCoherenceChecker
        │
        └─ Memory (FAISS + LTM JSON)
            ├─ PredictionErrorHistory
            ├─ SelfStateTracking
            ├─ PolicyTraces
            └─ LongTermMemory
```

Inter-layer messages flow through `GlobalWorkspace` only. Kinds: `vital_state`, `drive_signal`, `prediction_error`, `arousal_valence`, `goal`, `policy_proposal`, `metacognitive`, `world_model`.

---

## Project Structure

```
capstone/
├── .env                          # API keys (not committed)
├── .env.sample                   # Template
├── .python-version               # Pins Python 3.10.12 (required by animalai 5.0.1)
├── config.json                   # All tunable parameters
├── requirements.txt
├── run.sh                        # Entry point
├── animalai_configs/
│   └── basic_food.yaml           # Arena: food items + hazard walls
├── animal-ai-unity/              # Unity binary (built separately — see below)
│   └── AnimalAI.app
├── data/
│   └── long_term_memory/         # Persisted policy outcomes (JSON)
├── src/
│   └── core/
│       ├── main.py               # Entry point — loads config, wires everything
│       ├── adapters/
│       │   ├── base.py           # AbstractEnvironmentAdapter (ABC)
│       │   ├── loader.py         # Dynamic adapter loader (reads adapter_folder from config)
│       │   ├── animalai/         # Real Unity-backed adapter
│       │   │   ├── env_adapter.py
│       │   │   ├── homeostatic_wrapper.py
│       │   │   ├── observation_mapper.py
│       │   │   └── action_mapper.py
│       │   ├── headless/         # Pure Python simulation (no Unity needed)
│       │   │   └── env_adapter.py
│       │   └── iot/              # Stub — demonstrates extensibility
│       │       └── env_adapter.py
│       ├── coordination/
│       │   ├── messages.py       # AgentMessage dataclass
│       │   └── workspace.py      # GlobalWorkspace (publish / broadcast / clear)
│       ├── models/
│       │   ├── state.py          # AgentState, HomeostasisState, PerceptionState, …
│       │   ├── signals.py        # DriveChannel, DriveSignal, PredictionError, ArousalValence, …
│       │   └── memory_records.py # FAISS record types
│       ├── layers/
│       │   ├── interoceptive/
│       │   ├── predictive/
│       │   ├── action_selection/
│       │   └── metacognitive/
│       ├── memory/
│       │   ├── manager.py        # MemoryManager facade
│       │   ├── prediction_error_history.py
│       │   ├── self_state_tracking.py
│       │   ├── policy_traces.py
│       │   └── long_term_memory.py
│       ├── llm/
│       │   ├── factory.py        # build_llm_client() — dispatches on config["provider"]
│       │   └── providers/
│       │       ├── anthropic.py  # Full (default)
│       │       ├── openai.py     # Full
│       │       └── gemini.py     # Stub (NotImplementedError)
│       ├── observability/
│       │   └── logger.py         # RunLogger — JSONL telemetry per run
│       └── runtime/
│           └── loop.py           # AgentLoop — per-step cognitive cycle
└── tests/
    └── core/                     # 38 unit tests, no Unity required
```

---

## Setup

### Requirements

- **macOS** (tested on Apple Silicon and Intel)
- **pyenv** — manages the exact Python micro-version
- **Anthropic API key** (or OpenAI key if you prefer that provider)

### 1. Install pyenv (if not already installed)

```bash
brew install pyenv
# Add to your shell profile:
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Install Python 3.10.12

`animalai 5.0.1` requires Python `>=3.10.0,<3.10.13`. The project pins **3.10.12**.

```bash
pyenv install 3.10.12
```

The `.python-version` file at the project root automatically activates this version when you `cd` into the directory.

### 3. Clone and set up the project

```bash
git clone <repo-url>
cd capstone
```

### 4. Create the virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install grpcio==1.47.5 --only-binary=grpcio  # must pre-install; animalai pins <=1.48.2
pip install -r requirements.txt
```

> **Why the manual grpcio step?** `animalai` depends on `mlagents-envs` which requires
> `grpcio<=1.48.2`, and grpcio 1.48.2 has no pre-built wheel for Python 3.10 on macOS.
> 1.47.5 is the highest version with a binary wheel inside that range.

### 5. Create `.env`

```bash
cp .env.sample .env
# Edit .env and add your API key:
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Building Animal AI

Pre-built Animal AI binaries are no longer distributed (removed due to a Unity security advisory). You must build the Unity project yourself.

### Requirements

- **Unity Hub** — download from [unity.com/download](https://unity.com/download)
- **Unity 6000.0.23f1** — the exact editor version used by Animal AI v5

### Steps

**1. Install Unity Hub and add the editor**

Open Unity Hub → *Installs* → *Install Editor* → search for `6000.0.23f1` and install it.
Include the **Mac Build Support (Mono)** module if it is not pre-selected.

**2. Clone the Animal AI Unity project**

```bash
git clone https://github.com/Kinds-of-Intelligence-CFI/animal-ai-unity
```

**3. Open in Unity Hub**

*Projects* → *Add* → select the `animal-ai-unity` folder → open with Unity 6000.0.23f1.
Allow Unity to import assets (takes a few minutes on first open).

**4. Build for macOS**

`File → Build Settings` → select **macOS** → click **Build**.
Save the output as `AnimalAI.app` inside a folder at the project root, e.g.:

```
capstone/animal-ai-unity/AnimalAI.app
```

**5. Fix macOS permissions**

macOS quarantines newly built apps. Run once after every build:

```bash
xattr -r -d com.apple.quarantine ./animal-ai-unity/AnimalAI.app
chmod -R 755 ./animal-ai-unity/AnimalAI.app
```

**6. Verify the app opens**

```bash
open ./animal-ai-unity/AnimalAI.app
```

You should see a Unity splash screen / grey environment window. Close it before running the agent (the Python API launches its own instance).

---

## Connecting to Animal AI

### Option A — Python launches Unity automatically (recommended)

Set `file_name` in `config.json` to the path of the built app:

```json
"adapter_folder": "animalai",
"adapter_config": {
  "file_name": "./animal-ai-unity/AnimalAI.app",
  "arena_config": "animalai_configs/basic_food.yaml",
  "worker_id": 1,
  "base_port": 5005
}
```

Then just run:

```bash
./run.sh
```

The Python API will launch the Unity binary, wait for it to start, load the arena config, and begin stepping. The Unity window will appear in the background.

### Option B — Connect to a manually launched instance

If you want to control Unity separately (e.g., for visual debugging):

1. Launch the app manually: `open ./animal-ai-unity/AnimalAI.app`
2. Remove (or `null`) the `file_name` key in `config.json`
3. Run `./run.sh`

The Python API connects on port `base_port + worker_id` (default: **5006**).

### Troubleshooting connection failures

| Symptom | Fix |
|---------|-----|
| `TimeoutError` / no behavior specs | App not running, or wrong port. Check `worker_id` + `base_port`. |
| `Permission denied` on `.app` | Re-run the `xattr` + `chmod` commands above. |
| Port already in use | A previous crashed run may have left Unity alive: `lsof -i :5006 \| awk 'NR>1{print $2}' \| xargs kill -9` |
| Unity opens but immediately closes | Arena YAML is invalid — validate `animalai_configs/basic_food.yaml` |
| `ModuleNotFoundError: animalai` | Venv not activated or wrong Python. Run `source venv/bin/activate && python --version` — should print `3.10.12`. |

---

## Configuration

All parameters live in `config.json`. No code changes are needed to tune behaviour.

| Key | Description |
|-----|-------------|
| `adapter_folder` | `"animalai"` (Unity), `"headless"` (pure Python sim), `"iot"` (stub) |
| `adapter_config.file_name` | Path to Unity binary; omit to connect to a running instance |
| `adapter_config.arena_config` | Path to Animal AI YAML arena definition |
| `adapter_config.worker_id` | Worker index (allows multiple parallel environments) |
| `adapter_config.base_port` | Base port; Python connects on `base_port + worker_id` |
| `adapter_config.homeostatic.*` | Depletion rates, food restore amounts, hazard penalty |
| `llm.provider` | `"anthropic"` \| `"openai"` \| `"gemini"` (stub) |
| `llm.model` | Model name, e.g. `"claude-sonnet-4-6"` or `"gpt-4o-mini"` |
| `llm.enabled` | Set `false` to disable LLM entirely (urgency fallback only) |
| `policy_generator.llm_conflict_threshold` | Drive urgency gap that triggers LLM |
| `policy_generator.pe_streak_threshold` | Steps of high PE before LLM is called |
| `policy_generator.llm_reeval_interval` | Periodic LLM re-evaluation cadence (steps) |
| `memory.episode_length` | Steps per episode for LTM bookkeeping |
| `observability.log_root` | Directory for run logs |

### Environment variables (override config)

```bash
MAX_STEPS=200 ./run.sh          # override step count
LOG_PROMPTS=1 ./run.sh          # log every LLM prompt/response
LOG_STATE=0 ./run.sh            # disable AgentState snapshots
LOG_LEVEL=DEBUG ./run.sh        # verbose logging
```

---

## Running

```bash
source venv/bin/activate
./run.sh              # 500 steps (default)
./run.sh 200          # 200 steps
MAX_STEPS=1000 LOG_PROMPTS=1 ./run.sh
```

To use the headless simulation (no Unity needed):

```json
// config.json
"adapter_folder": "headless"
```

```bash
./run.sh
```

---

## Observability

Each run writes structured logs to `src/logs/runs/<timestamp>_<run_id>/`:

| File | Contents |
|------|----------|
| `run.json` | Metadata: config snapshot, Python version, adapter, start time |
| `metrics.jsonl` | Per-step: health, saturation, arousal, valence, PE, policy chosen |
| `events.jsonl` | Drive signals, PE summary, LLM trigger reason |
| `state.jsonl` | Full `AgentState` snapshots (gated by `LOG_STATE`) |
| `llm.jsonl` | Prompt + response + latency (gated by `LOG_PROMPTS`) |
| `memory.jsonl` | FAISS queries and updates (gated by `LOG_MEMORY`) |
| `tracebacks.jsonl` | Structured exception traces |

**Quick health plot:**

```bash
cat src/logs/runs/$(ls -t src/logs/runs | head -1)/metrics.jsonl \
  | python3 -c "import sys,json; [print(json.loads(l)['health']) for l in sys.stdin]"
```

---

## Swapping Environments

The brain has zero knowledge of the environment it runs in. To swap:

1. Change `"adapter_folder"` in `config.json`
2. That is all — no brain code changes

| `adapter_folder` | Environment |
|-----------------|-------------|
| `"animalai"` | Real Animal AI Unity testbed |
| `"headless"` | Pure Python 2D arena simulation (no Unity, no install) |
| `"iot"` | Stub — implement `src/core/adapters/iot/env_adapter.py` for real IoT sensors |

To add a new environment, create `src/core/adapters/<name>/env_adapter.py` implementing `AbstractEnvironmentAdapter` (see `src/core/adapters/base.py`) and add a `create_adapter(config)` factory function.

---

## LLM Provider

Change `config.json`:

```json
"llm": {
  "provider": "anthropic",   // or "openai"
  "model": "claude-sonnet-4-6"
}
```

Add the corresponding key to `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Set `"enabled": false` to run the brain on urgency-based heuristics only (no API calls).

The LLM is called **selectively** — only when one of 5 conditions is met:
1. Drive conflict (two channels both urgency > threshold)
2. Sustained prediction error streak
3. Skill gap (unfamiliar area, low policy confidence)
4. Periodic re-evaluation interval
5. Goal change detected

This prevents the rate-limiting and cost problems of calling the LLM every step.

---

## Testing

Tests run without Unity or any API keys:

```bash
source venv/bin/activate
PYTHONPATH=src pytest tests/ -v
```

38 tests cover: homeostatic wrapper, adapter contract, allostatic controller, prediction error calculator, policy generator (all 5 LLM trigger conditions), memory system (FAISS + LTM persistence), and full runtime loop.
