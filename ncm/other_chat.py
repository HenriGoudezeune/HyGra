"""ChatGPT and Gemini as chat partners next to Claude, through their own command-line agents and the
user's own (free) accounts, never an API key:
- ChatGPT: the Codex CLI (`codex exec --json`), logged in with ChatGPT.
- Gemini: the Antigravity CLI (`agy -p --output-format stream-json`), logged in with Google.

Each HyGra chat keeps one conversation per model (Codex thread id / Antigravity conversation id); the
next message resumes it. Like Claude in HyGra they get full access (no permission prompts). One process
per reply, read through a pipe: unlike Claude's jobs a reply stops when HyGra closes.
"""
import json
import os
import shutil
import subprocess
import threading

from ncm.claude_chat import (DATA, NO_WINDOW, SYSTEM, WORKDIR, result_line, stop, think_line,
                             tool_lines)

CODEX_CACHE = os.path.join(os.path.expanduser("~"), ".codex", "models_cache.json")
GEMINI_FALLBACK = "gemini-3.8-flash-high"
MAX_ARG = 24000   # the Windows command line holds 32767 chars; a longer Gemini prompt goes through a file

_procs = {}       # message id -> running Popen
_stopped = set()
_lock = threading.Lock()
_gemini = None    # (id, label) once asked


def _codex_model():
    """The model Codex itself recommends for this account (lowest priority among the listed ones)."""
    try:
        with open(CODEX_CACHE, encoding="utf-8") as f:
            ms = [m for m in json.load(f)["models"] if m.get("visibility") == "list"]
        m = min(ms, key=lambda m: m.get("priority", 99))
        return m["slug"], m.get("display_name") or m["slug"]
    except (OSError, ValueError, KeyError):
        return "gpt-6-luna", "GPT-6-Luna"


def _gemini_model():
    """The best Gemini the logged-in Google account offers: Pro (High) if it has it, else Flash (High)."""
    global _gemini
    if _gemini:
        return _gemini
    exe = shutil.which("agy")
    rows = []
    if exe:
        try:
            out = subprocess.run([exe, "models"], capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=30, creationflags=NO_WINDOW).stdout
            rows = [l.split("\t", 1) for l in out.splitlines() if l.startswith("gemini-") and "\t" in l]
        except (OSError, subprocess.SubprocessError):
            pass

    def rank(r):
        i = r[0]
        return ("pro" in i) * 4 + ("-high" in i) * 2 + ("-medium" in i)
    best = max(rows, key=rank) if rows else [GEMINI_FALLBACK, "Gemini 3.8 Flash (High)"]
    if rows:
        _gemini = (best[0], best[1].strip())
    return best[0], best[1].strip()


def models():
    """What the page shows in its model list: {key: {label, name, available}}."""
    out = {}
    if shutil.which("codex"):
        slug, name = _codex_model()
        out["gpt"] = {"label": f"ChatGPT {name} · gratis", "name": f"CHATGPT {name.upper()}", "model": slug}
    if shutil.which("agy"):
        gid, name = _gemini_model()
        out["gemini"] = {"label": f"{name} · Google", "name": name.upper(), "model": gid}
    return out


def _cwd(cwd):
    return cwd if cwd and os.path.isdir(cwd) else WORKDIR


def _codex(session, prompt):
    slug, _ = _codex_model()
    exe = shutil.which("codex")
    flags = ["--json", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
             "-m", slug, "-c", "model_reasoning_effort=high"]
    if session:
        return [exe, "exec", "resume", *flags, session, "-"], prompt
    return [exe, "exec", *flags, "-"], f"{SYSTEM}\n\n{prompt}"


def _agy(session, prompt, msg_id):
    gid, _ = _gemini_model()
    if not session:
        prompt = f"{SYSTEM}\n\n{prompt}"
    if len(prompt) > MAX_ARG:
        path = os.path.join(DATA, "prompts", f"{msg_id}.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(prompt)
        prompt = f"Mijn bericht is te lang voor de opdrachtregel en staat in {path}. Lees het en antwoord erop."
    cmd = [shutil.which("agy"), "--output-format", "stream-json", "--model", gid, "--dangerously-skip-permissions"]
    if session:
        cmd += ["--conversation", session]
    return cmd + ["--print=" + prompt], None


def run(kind, msg_id, session, prompt, cwd, on_text, on_status):
    """One reply, streamed through on_text; returns {text, session, tokens_in, tokens_out} or {error}."""
    cmd, stdin = _codex(session, prompt) if kind == "gpt" else _agy(session, prompt, msg_id)
    if not cmd[0]:
        return {"error": ("Codex" if kind == "gpt" else "Antigravity (agy)") + " is niet geïnstalleerd."}
    p = subprocess.Popen(cmd, cwd=_cwd(cwd), stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         errors="replace", creationflags=NO_WINDOW)
    with _lock:
        _procs[msg_id] = p
    err = []
    threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True).start()
    if stdin:
        try:
            p.stdin.write(stdin)
            p.stdin.close()
        except OSError:
            pass
    s = {"text": [], "session": session, "in": 0, "out": 0, "error": None}

    def emit(t):
        if t:
            s["text"].append(t)
            on_text(t)

    def para():   # a new piece of text after a tool call starts a new paragraph
        if s["text"] and not s["text"][-1].endswith("\n\n"):
            emit("\n\n")

    on_status("denkt na")
    try:
        for line in p.stdout:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            (_codex_event if kind == "gpt" else _agy_event)(e, s, emit, para, on_status)
        p.wait()
    finally:
        with _lock:
            _procs.pop(msg_id, None)
            was_stopped = msg_id in _stopped
            _stopped.discard(msg_id)
    if was_stopped:
        return {"stopped": True, "text": "".join(s["text"]), "session": s["session"]}
    text = "".join(s["text"]).strip()
    if s["error"] or (p.returncode and not text):
        tail = "".join(err).strip()[-2000:]
        return {"error": s["error"] or tail or f"gestopt met code {p.returncode}", "session": s["session"]}
    return {"text": text, "session": s["session"], "tokens_in": s["in"], "tokens_out": s["out"]}


def _codex_event(e, s, emit, para, on_status):
    t, item = e.get("type"), e.get("item") or {}
    if t == "thread.started":
        s["session"] = e.get("thread_id") or s["session"]
    elif t == "item.started" and item.get("type") not in ("agent_message", "reasoning"):
        on_status("werkt")
        emit(f"\n\n{_codex_tool(item)}\n\n")
    elif t == "item.completed":
        k = item.get("type")
        if k == "agent_message":
            on_status("schrijft")
            para()
            emit(item.get("text", ""))
        elif k == "reasoning" and (item.get("text") or "").strip():
            emit(f"\n\n{think_line(item['text'])}\n\n")
        elif k == "command_execution":
            emit(f"\n{result_line({'tool_use_id': item.get('id'), 'content': item.get('aggregated_output') or '', 'is_error': bool(item.get('exit_code'))})}\n\n")
        elif k not in ("agent_message", "reasoning"):
            emit(f"\n{result_line({'tool_use_id': item.get('id'), 'content': json.dumps({x: v for x, v in item.items() if x not in ('id', 'type')}, ensure_ascii=False)})}\n\n")
    elif t == "turn.completed":
        u = e.get("usage") or {}
        s["in"] += u.get("input_tokens") or 0
        s["out"] += (u.get("output_tokens") or 0) + (u.get("reasoning_output_tokens") or 0)
    elif t in ("turn.failed", "error"):
        s["error"] = (e.get("error") or {}).get("message") if isinstance(e.get("error"), dict) else e.get("message") or str(e)


def _codex_tool(item):
    k = item.get("type")
    if k == "command_execution":
        return tool_lines({"id": item.get("id"), "name": "Shell", "input": {"command": item.get("command", "")}})
    if k == "file_change":
        files = [c.get("path", "") for c in item.get("changes") or []]
        return tool_lines({"id": item.get("id"), "name": "Edit", "input": {"file_path": ", ".join(files)}})
    if k == "web_search":
        return tool_lines({"id": item.get("id"), "name": "WebSearch", "input": {"query": item.get("query", "")}})
    if k == "mcp_tool_call":
        return tool_lines({"id": item.get("id"), "name": f"mcp__{item.get('server')}__{item.get('tool')}",
                           "input": item.get("arguments") or {}})
    return tool_lines({"id": item.get("id"), "name": k or "tool", "input": {}})


def _agy_event(e, s, emit, para, on_status):
    if e.get("event") == "init":
        s["session"] = e.get("conversation_id") or s["session"]
    elif e.get("event") == "step_update":
        u = e["step_update"]
        s["session"] = u.get("conversation_id") or s["session"]
        typ, state = u.get("step_type"), u.get("state")
        if typ == "agent_response":
            if u.get("text_delta"):
                if not s.get("in_text"):
                    para()
                    s["in_text"] = True
                on_status("schrijft")
                emit(u["text_delta"])
            if state == "DONE":
                s["in_text"] = False
        elif typ == "tool":
            info = u.get("tool_info") or {}
            tid = f"s{u.get('step_index')}"
            inp = {k[0].lower() + k[1:]: v for k, v in (info.get("parameters") or {}).items()}
            if "commandLine" in inp:
                inp["command"] = inp.pop("commandLine")
            if state == "ACTIVE" and tid not in s.setdefault("tools", set()):   # once, even if it updates again
                s["tools"].add(tid)
                on_status("werkt")
                emit(f"\n\n{tool_lines({'id': tid, 'name': info.get('name') or u.get('tool_name'), 'input': inp})}\n\n")
            elif state in ("DONE", "ERROR"):
                emit(f"\n{result_line({'tool_use_id': tid, 'content': info.get('output') or '', 'is_error': state == 'ERROR'})}\n\n")
    elif e.get("event") == "result":
        r = e.get("result") or {}
        s["session"] = r.get("conversation_id") or s["session"]
        u = r.get("usage") or {}
        s["in"] = u.get("input_tokens") or 0
        s["out"] = (u.get("output_tokens") or 0)
        if r.get("status") != "SUCCESS":
            s["error"] = r.get("error") or r.get("status")
        elif not s["text"] and r.get("response"):
            emit(r["response"])


def stop_reply(msg_id):
    with _lock:
        p = _procs.get(msg_id)
        if p:
            _stopped.add(msg_id)
    if p:
        stop(p.pid)
