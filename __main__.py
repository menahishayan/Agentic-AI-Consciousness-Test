import json
import os
import time
from datetime import datetime
from pathlib import Path

import minedojo
import numpy as np
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
SECRETS_ENV = ROOT / ".env"
TELEMETRY_PATH = Path(
    os.getenv("TELEMETRY_PATH", str(ROOT / "telemetry.jsonl"))
).resolve()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TASK_ID = os.getenv("MINEDOJO_TASK_ID", "harvest_milk")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def summarize_obs(obs):
    summary = {}
    if isinstance(obs, dict):
        for k, v in obs.items():
            summary[k] = summarize_obs(v)
        return summary
    if isinstance(obs, np.ndarray):
        summary = {
            "shape": list(obs.shape),
            "dtype": str(obs.dtype),
        }
        if obs.size and np.issubdtype(obs.dtype, np.number):
            summary["min"] = float(np.nanmin(obs))
            summary["max"] = float(np.nanmax(obs))
        return summary
    if isinstance(obs, (list, tuple)):
        return {"type": type(obs).__name__, "len": len(obs)}
    if isinstance(obs, (int, float, str, bool)) or obs is None:
        return obs
    return {"type": type(obs).__name__}


def action_space_info(space):
    info = {"type": type(space).__name__}
    if hasattr(space, "n"):
        info["n"] = space.n
    if hasattr(space, "nvec"):
        info["nvec"] = list(space.nvec)
    if hasattr(space, "shape"):
        info["shape"] = list(space.shape)
    return info


def parse_action(text, space):
    try:
        data = json.loads(text)
        action = data.get("action", None) if isinstance(data, dict) else None
    except Exception:
        action = None

    if action is None:
        return space.sample(), {"parsed": False, "raw": text}

    if hasattr(space, "nvec"):
        arr = np.array(action, dtype=np.int64).reshape(-1)
        nvec = np.array(space.nvec, dtype=np.int64)
        if arr.size != nvec.size:
            return space.sample(), {"parsed": False, "raw": text}
        arr = np.clip(arr, 0, nvec - 1)
        return arr, {"parsed": True}

    if hasattr(space, "n") and isinstance(action, int):
        return int(max(0, min(space.n - 1, action))), {"parsed": True}

    if hasattr(space, "shape"):
        arr = np.array(action, dtype=np.float32).reshape(space.shape)
        if hasattr(space, "low") and hasattr(space, "high"):
            arr = np.clip(arr, space.low, space.high)
        return arr, {"parsed": True}

    return space.sample(), {"parsed": False, "raw": text}


def main():
    load_env_file(SECRETS_ENV)
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or export it.")

    client = OpenAI()
    env = minedojo.make(task_id=TASK_ID, image_size=(160, 256))

    task_prompt, task_guidance = minedojo.tasks.ALL_PROGRAMMATIC_TASK_INSTRUCTIONS[
        TASK_ID
    ]
    space = env.action_space
    space_info = action_space_info(space)

    obs = env.reset()
    TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TELEMETRY_PATH.open("a", encoding="utf-8") as f:
        for step in range(MAX_STEPS):
            obs_summary = summarize_obs(obs)
            prompt = (
                "You control a MineDojo agent.\n"
                f"Task: {task_prompt}\n"
                f"Guidance: {task_guidance}\n"
                f"Action space: {space_info}\n"
                f"Observation summary: {json.dumps(obs_summary)[:4000]}\n"
                'Return JSON only, like: {"action": [0,1,0,...]}.\n'
            )

            resp = client.responses.create(
                model=MODEL,
                input=prompt,
            )
            text = getattr(resp, "output_text", "") or ""
            action, parse_meta = parse_action(text, space)

            obs, reward, done, info = env.step(action)
            record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "step": step,
                "task_id": TASK_ID,
                "model": MODEL,
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "action_parse": parse_meta,
                "reward": reward,
                "done": done,
                "info_keys": list(info.keys()) if isinstance(info, dict) else None,
            }

            print(
                f"[step {step}] reward={reward} done={done} action={record['action']}"
            )
            f.write(json.dumps(record) + "\n")
            f.flush()

            if done:
                break
            time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    main()
