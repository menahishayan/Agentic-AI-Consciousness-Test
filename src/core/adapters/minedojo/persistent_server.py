"""
Persistent MineDojo environment server.

Run this ONCE and leave it running:
    PYTHONPATH=src python -m core.adapters.minedojo.persistent_server

It holds the Minecraft process alive and serves reset/step/close commands
over a local TCP socket (default port 9876).  Your main training loop
connects via RemoteMineDojoAdapter, which has the same interface as
MineDojoAdapter but talks to this server instead of owning the env.

Protocol  (all messages are length-prefixed pickled dicts):
  Client → Server:  {"cmd": "reset"}
                    {"cmd": "step", "action": <list>}
                    {"cmd": "close"}
                    {"cmd": "ping"}
  Server → Client:  {"ok": True,  "result": <payload>}
                    {"ok": False, "error": <str>}
"""
from __future__ import annotations

import argparse
import logging
import pickle
import socket
import struct
import traceback
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [persistent_server] %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 9876


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------

def _send(conn: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(data)) + data)


def _recv(conn: socket.socket) -> Any:
    raw_len = _recv_exactly(conn, 4)
    if not raw_len:
        raise EOFError("connection closed")
    (length,) = struct.unpack(">I", raw_len)
    return pickle.loads(_recv_exactly(conn, length))


def _recv_exactly(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("connection closed mid-message")
        buf += chunk
    return buf


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def _make_env(task_id: str, image_size: tuple) -> Any:
    import minedojo  # type: ignore
    log.info("Creating MineDojo env (task=%s, image_size=%s) …", task_id, image_size)
    env = minedojo.make(task_id=task_id, image_size=image_size)
    log.info("MineDojo env created.")
    return env


def _handle_client(conn: socket.socket, addr: Any, env: Any, noop: list) -> None:
    log.info("Client connected from %s", addr)
    try:
        while True:
            msg = _recv(conn)
            cmd = msg.get("cmd")

            if cmd == "ping":
                _send(conn, {"ok": True, "result": "pong"})

            elif cmd == "reset":
                try:
                    result = env.reset()
                    if isinstance(result, tuple) and len(result) == 2:
                        obs, info = result
                    else:
                        obs, info = result, {}
                    if not isinstance(info, dict):
                        info = {}
                    _send(conn, {"ok": True, "result": (obs, info)})
                    log.info("reset() OK")
                except Exception as exc:
                    log.exception("reset() failed")
                    _send(conn, {"ok": False, "error": str(exc)})

            elif cmd == "step":
                action = msg.get("action", noop)
                try:
                    result = env.step(action)
                    if isinstance(result, tuple) and len(result) == 5:
                        obs, reward, terminated, truncated, info = result
                        done = bool(terminated or truncated)
                    else:
                        obs, reward, done, info = result
                    if not isinstance(info, dict):
                        info = {}
                    _send(conn, {"ok": True, "result": (obs, reward, done, info)})
                except ValueError as exc:
                    # Illegal action (e.g. place-air) — silently fall back to noop
                    log.warning("Illegal action, falling back to noop: %s", exc)
                    try:
                        result = env.step(noop)
                        if isinstance(result, tuple) and len(result) == 5:
                            obs, reward, terminated, truncated, info = result
                            done = bool(terminated or truncated)
                        else:
                            obs, reward, done, info = result
                        if not isinstance(info, dict):
                            info = {}
                        _send(conn, {"ok": True, "result": (obs, reward, done, info)})
                    except Exception as inner:
                        log.exception("noop fallback also failed")
                        _send(conn, {"ok": False, "error": str(inner)})
                except Exception as exc:
                    log.exception("step() failed")
                    _send(conn, {"ok": False, "error": str(exc)})

            elif cmd == "close":
                log.info("Client requested close (env stays alive)")
                _send(conn, {"ok": True, "result": None})
                break

            elif cmd == "get_action_space":
                space = getattr(env, "action_space", None)
                nvec = None
                noop_vec = noop
                if space is not None and hasattr(space, "nvec"):
                    nvec = [int(v) for v in space.nvec]
                if space is not None and hasattr(space, "no_op"):
                    noop_vec = [int(v) for v in list(space.no_op())]
                _send(conn, {"ok": True, "result": {"nvec": nvec, "noop": noop_vec}})

            else:
                _send(conn, {"ok": False, "error": f"unknown command: {cmd!r}"})

    except EOFError:
        log.info("Client %s disconnected.", addr)
    except Exception:
        log.exception("Unhandled error handling client %s", addr)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _build_noop(env: Any) -> list:
    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "no_op"):
        return [int(v) for v in list(space.no_op())]
    if space is not None and hasattr(space, "sample"):
        sample = space.sample()
        return [0] * len(sample)
    return [0] * 8


def serve(host: str, port: int, task_id: str, image_size: tuple) -> None:
    env = _make_env(task_id, image_size)
    noop = _build_noop(env)
    log.info("Noop action: %s", noop)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)  # one client at a time (your training loop is sequential)
    log.info("Listening on %s:%d  — Minecraft is ready, connect anytime.", host, port)

    try:
        while True:
            conn, addr = srv.accept()
            _handle_client(conn, addr, env, noop)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        srv.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Persistent MineDojo env server")
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--task-id", default="harvest_milk")
    parser.add_argument("--image-size", default="160,256")
    args = parser.parse_args()
    h, w = (int(x) for x in args.image_size.split(","))
    serve(args.host, args.port, args.task_id, (h, w))
