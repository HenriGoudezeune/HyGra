"""ChatGPT and Gemini as chat partners next to Claude, through their own command-line agents and the
user's own (free) accounts, never an API key:
- ChatGPT: the Codex CLI (`codex exec --json`), logged in with ChatGPT.
- Gemini: the Antigravity CLI (`agy -p --output-format stream-json`), logged in with Google.
- a local coding model (Qwen3-Coder 30B) through Ollama: no account, nothing leaves the laptop, no tools.

Each HyGra chat keeps one conversation per model (Codex thread id / Antigravity conversation id); the
next message resumes it. Like Claude in HyGra they get full access (no permission prompts). One process
per reply, read through a pipe: unlike Claude's jobs a reply stops when HyGra closes.
"""
import glob
import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import time
from datetime import datetime

from ncm.claude_chat import (DATA, NO_WINDOW, SYSTEM, WORKDIR, result_line, stop, think_line,
                             tool_lines)

CODEX_CACHE = os.path.join(os.path.expanduser("~"), ".codex", "models_cache.json")
GEMINI_FALLBACK = "gemini-3.8-flash-high"
MAX_ARG = 24000   # the Windows command line holds 32767 chars; a longer Gemini prompt goes through a file

_procs = {}       # message id -> running Popen
_stopped = set()
_lock = threading.Lock()
_rows = None      # `agy models` rows once asked


def _codex_model():
    """The model Codex itself recommends for this account (lowest priority among the listed ones)."""
    try:
        with open(CODEX_CACHE, encoding="utf-8") as f:
            ms = [m for m in json.load(f)["models"] if m.get("visibility") == "list"]
        m = min(ms, key=lambda m: m.get("priority", 99))
        return m["slug"], m.get("display_name") or m["slug"]
    except (OSError, ValueError, KeyError):
        return "gpt-6-luna", "GPT-6-Luna"


def gemini_rows(fresh=False):
    """The Gemini models the logged-in Google account offers ([id, label] rows); empty when not logged in."""
    global _rows
    if _rows is not None and not fresh:
        return _rows
    exe = shutil.which("agy")
    rows = []
    if exe:
        try:
            out = subprocess.run([exe, "models"], capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=30, creationflags=NO_WINDOW).stdout
            rows = [l.split("	", 1) for l in out.splitlines() if l.startswith("gemini-") and "	" in l]
        except (OSError, subprocess.SubprocessError):
            pass
    _rows = rows
    return rows


def gemini_model():
    """The best Gemini the logged-in Google account offers: Pro (High) if it has it, else Flash (High)."""
    rows = gemini_rows()

    def rank(r):
        i = r[0]
        return ("pro" in i) * 4 + ("-high" in i) * 2 + ("-medium" in i)
    best = max(rows, key=rank) if rows else [GEMINI_FALLBACK, "Gemini 3.8 Flash (High)"]
    return best[0], best[1].strip()


def models():
    """What the page shows in its model list: {key: {label, name, available}}."""
    out = {}
    if shutil.which("codex"):
        slug, name = _codex_model()
        out["gpt"] = {"label": f"ChatGPT {name} · gratis", "name": f"CHATGPT {name.upper()}", "model": slug}
    if shutil.which("agy"):
        gid, name = gemini_model()
        out["gemini"] = {"label": f"{name} · Google", "name": name.upper(), "model": gid}
    out["local"] = {"label": f"{LOCAL_NAME} · lokaal", "name": LOCAL_NAME.upper(), "model": LOCAL_MODEL}
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
    gid, _ = gemini_model()
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
    if p is None:
        return
    if hasattr(p, "pid"):
        stop(p.pid)
    else:   # a local reply: closing the stream ends the generation in Ollama too
        try:
            p.close()
        except OSError:
            pass


# ----- usage: how much of each account's limit is used, as {windows: [{k, label, used 0..1, resets unix}], at}
def _window_label(minutes):
    return {300: "5 UUR", 1440: "DAG", 10080: "WEEK", 43200: "30 DAGEN"}.get(minutes) or (
        f"{minutes // 1440} DAGEN" if minutes % 1440 == 0 else f"{minutes // 60} UUR")


def _save_limits(kind, out):
    if out.get("windows"):
        out["at"] = time.time()
        os.makedirs(DATA, exist_ok=True)
        with open(os.path.join(DATA, f"limits_{kind}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f)
    return out


def last_limits(kind):
    try:
        with open(os.path.join(DATA, f"limits_{kind}.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def codex_limits(thread=None):
    """Codex writes the ChatGPT plan's rate limits into its session files after every turn: read the newest."""
    files = glob.glob(os.path.join(os.path.expanduser("~"), ".codex", "sessions", "*", "*", "*",
                                   f"rollout-*{thread or ''}.jsonl"))
    for path in sorted(files, key=os.path.getmtime, reverse=True)[:5]:
        last = None
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"rate_limits"' in line:
                    last = line
        if not last:
            continue
        rl = (json.loads(last).get("payload") or {}).get("rate_limits") or {}
        wins = [{"k": k, "label": _window_label(w.get("window_minutes") or 0), "used": (w.get("used_percent") or 0) / 100,
                 "resets": w.get("resets_at")} for k in ("primary", "secondary") if (w := rl.get(k))]
        return _save_limits("gpt", {"windows": wins, "plan": (rl.get("plan_type") or "").upper()})
    return last_limits("gpt")


def gemini_limits():
    """`agy -p /usage` answers without a model call: the Google account's quota per model group."""
    exe = shutil.which("agy")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--output-format", "json", "--print=/usage"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30, creationflags=NO_WINDOW).stdout
        data = json.loads(out)["command"]["data"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return last_limits("gemini")
    wins = []
    for g in data.get("groups") or []:
        if "gemini" not in (g.get("name") or "").lower():
            continue
        for b in g.get("buckets") or []:
            try:
                resets = datetime.fromisoformat(b["reset_time"].replace("Z", "+00:00")).timestamp()
            except (KeyError, ValueError):
                resets = None
            wins.append({"k": b.get("id"), "label": {"weekly": "WEEK", "daily": "DAG"}.get(b.get("window"), (b.get("window") or "").upper()),
                         "used": 1 - (b.get("remaining_fraction") if b.get("remaining_fraction") is not None else 1), "resets": resets})
    return _save_limits("gemini", {"windows": wins})


def limits(kind, refresh=False):
    if not refresh:
        return last_limits(kind)
    return codex_limits() if kind == "gpt" else gemini_limits()


# ----- a local coding model through Ollama (http://127.0.0.1:11434): free, offline, no account
OLLAMA = "http://127.0.0.1:11434"
LOCAL_MODEL = "qwen3-coder:30b"   # 30B mixture-of-experts, ~3B active per token: fits 4 GB VRAM + RAM at usable speed
LOCAL_NAME = "Qwen3-Coder 30B"
LOCAL_CTX = 16384


def ollama_exe():
    return shutil.which("ollama") or next((p for p in [os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")]
                                           if os.path.exists(p)), None)


def _ollama(path, body=None, timeout=5):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def ollama_up(start=True):
    """True once the Ollama server answers; starts `ollama serve` in the background if it is installed but not running."""
    try:
        _ollama("/api/version", timeout=1).read()
        return True
    except OSError:
        pass
    exe = ollama_exe()
    if not (start and exe):
        return False
    subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    for _ in range(40):
        time.sleep(0.25)
        try:
            _ollama("/api/version", timeout=1).read()
            return True
        except OSError:
            pass
    return False


def local_ready():
    """Whether the model is downloaded (needs the server)."""
    if not ollama_up():
        return False
    try:
        tags = json.load(_ollama("/api/tags"))
    except (OSError, ValueError):
        return False
    return any(m.get("name") == LOCAL_MODEL or m.get("model") == LOCAL_MODEL for m in tags.get("models") or [])


def run_local(msg_id, history, prompt, on_text, on_status):
    """One reply from the local model. It has no memory of its own: history ([{role, content}]) is the chat so far."""
    if not ollama_up():
        return {"error": "Ollama draait niet en kon niet gestart worden. Installeer het via MODELLEN BEHEREN."}
    msgs = [{"role": "system", "content": SYSTEM}] + list(history or [])[-40:] + [{"role": "user", "content": prompt}]
    on_status("denkt na")   # the first reply also loads ~19 GB from disk into memory
    body = {"model": LOCAL_MODEL, "messages": msgs, "stream": True, "options": {"num_ctx": LOCAL_CTX}}
    text, last = [], {}
    try:
        r = _ollama("/api/chat", body, timeout=600)
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode(errors="replace") or str(e)}
    except OSError as e:
        return {"error": str(e)}
    with _lock:
        _procs[msg_id] = r
    try:
        for line in r:
            if msg_id in _stopped:
                break
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("error"):
                return {"error": e["error"]}
            piece = (e.get("message") or {}).get("content") or ""
            if piece:
                if not text:
                    on_status("schrijft")
                text.append(piece)
                on_text(piece)
            if e.get("done"):
                last = e
    except OSError:
        pass   # closed by stop_reply
    finally:
        r.close()
        with _lock:
            _procs.pop(msg_id, None)
            was_stopped = msg_id in _stopped
            _stopped.discard(msg_id)
    if was_stopped:
        return {"stopped": True, "text": "".join(text)}
    out = {"text": "".join(text).strip(), "tokens_in": last.get("prompt_eval_count") or 0, "tokens_out": last.get("eval_count") or 0}
    if last.get("eval_duration"):
        out["tok_s"] = round(out["tokens_out"] / (last["eval_duration"] / 1e9), 1)
    return out
