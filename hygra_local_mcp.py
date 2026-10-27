"""hygra-local: an MCP server (stdio) that lets Claude Code hand small coding tasks to the local model that
HyGra also uses (Qwen3-Coder 30B through Ollama). Free, offline, and several tasks can run side by side, so
an agent can farm out drafts while it keeps working, and subagents can each use it too.

Register once (user scope, so every project has it):
    claude mcp add --scope user hygra-local -- <python> <path to this file>
Standard library only: it talks JSON-RPC on stdin/stdout and HTTP to Ollama.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncm import other_chat  # noqa: E402

MAX_FILE = 40000      # characters per attached file
MAX_PARALLEL = 4      # Ollama queues anything beyond what it runs at once
SYSTEM = ("You are a careful senior programmer working as a helper for another AI agent. Do exactly the task given, "
          "nothing more. Reply with the result only: code in fenced blocks with the language, at most a few short "
          "lines of explanation when they are needed. If the task is unclear or impossible, say so in one line.")

TOOLS = [
    {
        "name": "local_code",
        "description": (
            "Hand ONE small, self-contained coding task to a local model (Qwen3-Coder 30B on this laptop via Ollama). "
            "Free and it uses none of the user's Claude plan, but it is slower than you (roughly 5-15 tokens/s; the "
            "first call after a while also loads the model, up to a minute) and weaker: about the level of a good "
            "mid-2025 coding model, well below Claude. Good for: boilerplate, a single well-specified function, unit "
            "tests for given code, docstrings/comments, type hints, converting code between languages or formats, "
            "regexes, small refactors of one file, explaining a snippet. Not for: tasks that need tools, running code, "
            "the wider repository, or subtle design judgment. It cannot see files unless you pass them in `files`. "
            "Treat its answer as an untrusted draft: review or test it before using it."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What to do, fully specified: inputs, expected output, language, constraints."},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Absolute paths of files whose contents the model needs."},
                "max_tokens": {"type": "integer", "description": "Upper bound on the answer length (default 2048)."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "local_code_batch",
        "description": (
            "Run SEVERAL independent small coding tasks on the local model (Qwen3-Coder 30B) at the same time and get "
            "all answers back together. Same strengths and limits as local_code. Use it to fan out e.g. tests for "
            "several functions, docstrings for several files, or alternative drafts, while you keep the hard parts. "
            "Tasks share the laptop's GPU/CPU, so 2-4 at once is the sweet spot; more just queue."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"task": {"type": "string"}, "files": {"type": "array", "items": {"type": "string"}}},
                        "required": ["task"],
                    },
                },
                "max_tokens": {"type": "integer", "description": "Upper bound per answer (default 2048)."},
            },
            "required": ["tasks"],
        },
    },
]


def _prompt(task, files):
    parts = [task]
    for f in files or []:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                body = fh.read()
        except OSError as e:
            parts.append(f"\n[{f}: cannot read: {e}]")
            continue
        if len(body) > MAX_FILE:
            body = body[:MAX_FILE] + f"\n... ({len(body) - MAX_FILE} more characters cut)"
        parts.append(f"\n--- {f} ---\n{body}")
    return "\n".join(parts)


def ask(task, files=None, max_tokens=2048):
    if not other_chat.ollama_up():
        return "ERROR: Ollama is not running and could not be started (install it from HyGra's MODELLEN panel).", True
    body = {"model": other_chat.LOCAL_MODEL, "stream": False,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _prompt(task, files)}],
            "options": {"num_ctx": other_chat.LOCAL_CTX, "num_predict": max_tokens}}
    t0 = time.time()
    try:
        r = json.load(other_chat._ollama("/api/chat", body, timeout=900))
    except Exception as e:  # noqa: BLE001  (reported to the calling agent, never raised)
        return f"ERROR: {e}", True
    if r.get("error"):
        return f"ERROR: {r['error']}", True
    n, dur = r.get("eval_count") or 0, (r.get("eval_duration") or 0) / 1e9
    speed = f", {n / dur:.1f} tok/s" if dur else ""
    return f"{(r.get('message') or {}).get('content', '').strip()}\n\n[{other_chat.LOCAL_NAME}, local: {time.time() - t0:.0f} s{speed}]", False


def call(name, args):
    mt = int(args.get("max_tokens") or 2048)
    if name == "local_code":
        return ask(args["task"], args.get("files"), mt)
    if name == "local_code_batch":
        tasks = args.get("tasks") or []
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, max(1, len(tasks)))) as pool:
            res = list(pool.map(lambda t: ask(t["task"], t.get("files"), mt), tasks))
        text = "\n\n".join(f"=== task {i + 1} ===\n{out}" for i, (out, _) in enumerate(res))
        return text, all(err for _, err in res) and bool(res)
    return f"ERROR: unknown tool {name}", True


_out = threading.Lock()


def send(msg):
    with _out:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()


def handle(m):
    mid, method = m.get("id"), m.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": (m.get("params") or {}).get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}}, "serverInfo": {"name": "hygra-local", "version": "1.0"}}})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        p = m.get("params") or {}
        text, err = call(p.get("name"), p.get("arguments") or {})
        send({"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}], "isError": err}})
    elif method == "ping":
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
    elif mid is not None:   # requests we do not know; notifications need no answer
        send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}})


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        # every request on its own thread: parallel tool calls from Claude really run side by side
        threading.Thread(target=handle, args=(m,), daemon=True).start()


if __name__ == "__main__":
    main()
