# Capstone — Biomimetic Agent (MineDojo)

A biomimetic AI agent grounded in predictive processing and homeostatic drive-regulation, evaluated inside the MineDojo/Minecraft environment. The architecture implements five functional layers (interoceptive, perceptual, metacognitive, action-selection, memory) communicating through a shared `GlobalWorkspace`. All configuration is centralized in `config.json`; all runtime state is logged per-run to `src/logs/runs/`.

---

## Prerequisites

| Dependency | Version |
|---|---|
| Python | 3.9.x (required — MineDojo's Malmo build is incompatible with 3.10+) |
| JDK | 8 (Temurin recommended) |
| pip / setuptools / wheel | Pinned — see below |
| NumPy | `<2` (MineDojo references `np.unicode_`, removed in NumPy 2) |
| OpenCV | `4.8.1.78` (pinned for NumPy `<2` compatibility on Python 3.9) |
| gym | `0.21.0` (requires legacy pip toolchain to build correctly) |

---

## Setup

### 1. Install Rosetta and JDK 8

```bash
softwareupdate --install-rosetta --agree-to-license
brew install --cask temurin@8
```

Set `JAVA_HOME` before running anything:

```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
```

### 2. Create a Python 3.9 virtual environment

```bash
brew install pyenv
pyenv install 3.9.25
~/.pyenv/versions/3.9.25/bin/python -m venv ./.venv
```

### 3. Install dependencies

Downgrade pip toolchain first — gym 0.21.0 fails to build with newer versions:

```bash
./.venv/bin/python -m pip install "pip<23" "setuptools<65" "wheel<0.40" "packaging<23"
./.venv/bin/python -m pip install -r ./requirements.txt
./.venv/bin/python -m pip install -e ./MineDojo
```

### 4. Apply the Gradle patch

The stock MineDojo build references a JitPack SHA that no longer resolves. The patch switches to the SpongePowered repo and pins `org.spongepowered:mixingradle:0.6-SNAPSHOT`:

```bash
git -C ./MineDojo apply ./gradle-fix.patch
```

### 5. Validate

```bash
./.venv/bin/python ./MineDojo/scripts/validate_install.py
```

---

## Environment variables

Create a `.env` file at the project root. `core/main.py` loads it automatically via `load_env_file`.

```dotenv
# LLM provider keys (only needed when llm.enabled is true in config.json)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...

# Runtime options
MAX_STEPS=50               # Steps per run (default: 50)
INCLUDE_INVENTORY=1        # Log inventory in AgentState (default: 1)
INCLUDE_VOXELS=1           # Log voxel data in AgentState (default: 1)

# Observability
LOG_ROOT=src/logs/runs     # Root directory for per-run logs
LOG_PROMPTS=0              # Log full LLM prompts (default: 0)
LOG_MEMORY=1               # Log memory operations (default: 1)
LOG_STATE=1                # Log AgentState each step (default: 1)
```

---

## Running the agent

### Option A — Persistent server (recommended)

The persistent server keeps the Minecraft process alive across multiple agent runs, avoiding the ~60-second JVM startup on every reset. Start it once and leave it running:

```bash
JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home \
PYTHONPATH=src \
./.venv/bin/python -m core.adapters.minedojo.persistent_server
```

The server listens on `127.0.0.1:9876` by default. Then run the agent:

```bash
JAVA_HOME=... PYTHONPATH=src ./.venv/bin/python -m core.main
```

Set `"use_remote": true` in `config.json` (under `adapter_config`) so the agent connects to the server rather than spawning its own process.

### Option B — Standalone (spawns its own Minecraft process)

Set `"use_remote": false` (or omit it) in `config.json`. The agent owns the full environment lifecycle.

```bash
JAVA_HOME=... PYTHONPATH=src ./.venv/bin/python -m core.main
```

---

## Configuration (`config.json`)

`config.json` lives at the project root. All keys have defaults baked into `core/main.py`; the file deep-merges on top of those defaults, so only overrides need to be specified.

### Full annotated schema

```jsonc
{
  // Which adapter subfolder to load from src/core/adapters/
  "adapter_folder": "minedojo",

  // LLM integration (currently optional — set enabled: false to skip)
  "llm": {
    "enabled": false,
    "provider": "openai",           // "openai" | "anthropic" | "gemini"
    "model": "gpt-4.1-mini",
    "temperature": 0.1,
    "max_tokens": 500,
    "timeout_s": 1.0
  },

  // Passed directly to the adapter on construction
  "adapter_config": {
    "task_id": "harvest_milk",
    "image_size": [160, 256],
    "initial_health": 6,            // Starting health (out of 20)
    "initial_food": 4,              // Starting food (out of 20)
    "use_remote": true,             // Connect to persistent_server instead of spawning
    "remote_host": "127.0.0.1",    // Optional; default shown
    "remote_port": 9876             // Optional; default shown
  },

  "policy_generator": {

    // Scoring weights for policy selection
    // goal_coherence + prediction_error must sum to <= 1.0
    "weights": {
      "goal_coherence": 0.6,
      "prediction_error": 0.4,
      "allostatic_survival_fit": 0.2,
      "allostatic_urgency_alignment": 0.2
    },

    // Fallback scores used when a scorer cannot produce a value
    "fallback_scores": {
      "goal_coherence": 0.5,
      "prediction_error": 0.5,
      "allostatic_survival_fit": 0.5,
      "allostatic_urgency_alignment": 0.5
    },

    // Long-term memory (persistent policy performance history)
    "long_term_memory": {
      "path": "data/long_term_memory",
      "max_score_history": 200,
      "max_outcome_history": 200
    },

    // Policy-level prediction error
    "max_expected_error": 1.0,
    "prediction_error_window": 20,

    // Allostatic / homeostatic controller
    "allostatic_controller": {
      "planning_horizon": 50,
      "history_window": 20,
      "irreversibility_bonus": 0.3,   // Extra weight for irreversible drive violations
      "recovery_weight_factor": 0.2,
      "urgency_tie_epsilon": 0.05,
      "threat_prior_weight": 0.3,
      "min_confidence": 0.5,

      // Drive channels — each maps to a homeostatic variable
      "channels": [
        {
          "id": "health",
          "setpoint": 0.9,
          "critical_threshold": 0.25,
          "irreversible": true,
          "recovery_cost_ticks": 25,
          "suggested_action_tag": "heal"
        },
        { "id": "hunger",   "setpoint": 0.8, "critical_threshold": 0.2,  "irreversible": false, "recovery_cost_ticks": 20, "suggested_action_tag": "eat"     },
        { "id": "oxygen",   "setpoint": 0.9, "critical_threshold": 0.2,  "irreversible": true,  "recovery_cost_ticks": 30, "suggested_action_tag": "surface"  },
        { "id": "resource_level", "setpoint": 0.7, "critical_threshold": 0.3, "irreversible": false, "recovery_cost_ticks": 15, "suggested_action_tag": "gather" },
        { "id": "safety",   "setpoint": 0.8, "critical_threshold": 0.35, "irreversible": true,  "recovery_cost_ticks": 20, "suggested_action_tag": "retreat"  }
      ]
    },

    // Perceptual prediction error (per-feature EMA)
    "perceptual_prediction_error": {
      "alpha": 0.1,
      "epsilon": 0.01,
      "sigma_clip": 3.0,
      "default_precision": 0.5,
      "min_precision": 0.3
    },

    // Working-memory and FAISS retrieval settings
    "memory": {
      "working_memory_capacity": 100,
      "pe_min_observations": 5,
      "pe_ema_alpha": 0.1,
      "faiss_k_default": 5,
      "faiss_epsilon": 1e-6,
      "episode_length": 1000
    },

    // Arousal/valence system weights
    "arousal_valence": {
      "w_health": 0.35,
      "w_hunger": 0.25,
      "w_threat": 0.30,
      "w_pred_err": 0.10,
      "v_health": 0.30,
      "v_hunger": 0.30,
      "v_resources": 0.20,
      "v_oxygen": 0.20,
      "decay_rate": 0.95,
      "resting_arousal": 0.10,
      "urgency_broadcast_threshold": 0.65
    }
  }
}
```

---

## Adapter contract

Adapters live in `src/core/adapters/<adapter_folder>/`. The loader at `src/core/adapters/loader.py` resolves the folder name from config and constructs the adapter.

**Required methods:**

| Method | Signature | Notes |
|---|---|---|
| `reset` | `() -> (obs, info)` | |
| `step` | `(action) -> (obs, reward, done, info)` | |
| `close` | `() -> None` | |
| `get_available_vitals` | `() -> list[str]` | Returns homeostatic variable names |
| `get_available_policies` | `() -> list[dict]` | Each entry must have `tags` and `drive_tags` |
| `estimate_resource_level` | `(...) -> float` | |
| `estimate_threat_proximity` | `(...) -> float` | |
| `build_area_id` | `(...) -> str` | |

**Optional methods:**

| Method | Notes |
|---|---|
| `sample_action()` | Random action for exploration |
| `estimate_entity_density(...)` | |
| `estimate_terrain_novelty(...)` | |

Policy descriptor requirements: each dict returned by `get_available_policies()` must include non-empty `tags` and `drive_tags`. Drive tags may be inferred from `tags`, `description`, or the callable name if omitted, but explicit tags are preferred.

---

## Memory subsystems

`MemoryManager` (`src/core/memory/manager.py`) owns four sub-systems:

| Sub-system | Class | Purpose |
|---|---|---|
| Working Memory Buffer | `WorkingMemoryBuffer` | Ring buffer of the most recent N observations/states |
| Prediction Error History | `PredictionErrorHistory` | EMA-tracked per-feature prediction errors with FAISS retrieval |
| Self-State Tracking | `SelfStateTracking` | Longitudinal homeostatic variable history with FAISS retrieval |
| Policy Traces | `PolicyTraces` | Per-episode policy selection and outcome records with FAISS retrieval |

Long-term memory is a separate persistent layer (`LongTermMemory`) backed by JSON files in `src/data/long_term_memory/policies/`. It survives restarts and accumulates performance history across runs.

---

## AgentState

`AgentState` (`src/core/models/state.py`) is built from the MineDojo `info` dict each step and passed between modules.

```python
from core.models.state import AgentState

state = AgentState.from_info(info)
state_dict = state.to_dict(include_inventory=False, include_voxels=False)
```

**Homeostasis fields:** `life`, `armor`, `food`, `saturation`, `xp`, `air`, `is_sleeping`, `is_alive`, `is_dead`

**Position/orientation:** `xpos`, `ypos`, `zpos`, `pitch`, `yaw`

**Biome/world:** `biome_name`, `biome_id`, `biome_temperature`, `biome_rainfall`, `sea_level`

**Lighting/weather:** `light_level`, `sky_light_level`, `sun_brightness`, `is_raining`, `can_see_sky`

**Time:** `world_time`, `total_time`

**Local structures:** `nearby_furnace`, `nearby_crafting_table`

**Heavy fields (opt-in via env vars):** `voxels` (`INCLUDE_VOXELS=1`), `inventory`, `inventories_available`, `current_item_index` (`INCLUDE_INVENTORY=1`)

**Misc:** `distance_travelled_cm`, `stat`, `achievement`, `damage_source`, `score`, `name`

---

## Observability and logging

Each run produces an isolated directory under `src/logs/runs/<timestamp>_<run_id>/` containing:

| File | Contents |
|---|---|
| `run.json` | Run metadata (start time, PID, Python version, full config) |
| `events.jsonl` | Step-level events (policy selected, drive signals, scores) |
| `state.jsonl` | AgentState snapshot each step (gated by `LOG_STATE`) |
| `llm.jsonl` | LLM prompt/response pairs (gated by `LOG_PROMPTS`) |
| `memory.jsonl` | Memory read/write operations (gated by `LOG_MEMORY`) |
| `tracebacks.jsonl` | Structured exception records |
| `tracebacks.log` | Human-readable traceback text |

Watchdog logs are written separately to `src/logs/watchdog/`.

> **Note:** The JSONL telemetry currently logs `info_keys` (available field names), not their values. To log values, update `__main__.py` to include selected `info` and `obs` fields in each record.

---

## Project structure

```
.
├── config.json                         # Runtime configuration (deep-merged with defaults)
├── requirements.txt
├── MineDojo/                           # MineDojo submodule + gradle-fix.patch
└── src/
    ├── core/
    │   ├── main.py                     # Entry point — loads config, builds components, starts AgentLoop
    │   ├── adapters/
    │   │   ├── loader.py
    │   │   └── minedojo/
    │   │       ├── env_adapter.py      # MineDojoAdapter (owns the Minecraft process)
    │   │       ├── remote_adapter.py   # RemoteMineDojoAdapter (connects to persistent_server)
    │   │       ├── persistent_server.py# Long-lived server — keeps Minecraft alive between runs
    │   │       ├── action_mapper.py
    │   │       ├── observation_mapper.py
    │   │       └── skill_library/      # Voyager-style pre-built policy descriptors
    │   ├── layers/
    │   │   ├── interoceptive/          # VitalStateMonitor, AllostaticController, ArousalValenceSystem
    │   │   ├── action_selection/       # PolicyGenerator, FreeEnergyMinimizer, MotorControlInterface
    │   │   ├── predictive.py           # Policy-level PredictionErrorCalculator
    │   │   └── metacognitive.py        # GoalCoherenceChecker
    │   ├── memory/
    │   │   ├── manager.py              # MemoryManager — owns all four sub-systems
    │   │   ├── working_memory_buffer.py
    │   │   ├── prediction_error_history.py
    │   │   ├── self_state_tracking.py
    │   │   ├── policy_traces.py
    │   │   └── long_term_memory.py     # Persistent JSON-backed policy history
    │   ├── perceptual/                 # Perceptual PredictionErrorCalculator
    │   ├── runtime/
    │   │   └── loop.py                 # AgentLoop — main per-step control flow
    │   ├── coordination/               # GlobalWorkspace, AgentMessage
    │   ├── models/                     # AgentState, ActionProposal, signal types
    │   ├── llm/                        # Provider-agnostic LLM client (OpenAI / Anthropic / Gemini)
    │   └── observability/              # RunLogger, LoggingConfig, paths, serializer
    ├── data/
    │   └── long_term_memory/
    │       └── policies/               # Persistent per-policy JSON records (persists across runs)
    └── logs/
        ├── runs/                       # Per-run structured logs
        └── watchdog/                   # Watchdog process logs
```
