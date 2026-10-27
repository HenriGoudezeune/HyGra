"""Claude Opus 5.5 as a chat partner in the app, through the local Claude Code CLI.

The CLI is logged in with the user's Claude subscription (Pro/Max), so chatting costs
nothing extra and counts toward the plan's normal usage limits; no API key is needed.
Each HyGra chat is one Claude Code session (`--session-id` on the first message,
`--resume` afterwards) with full access, like Claude Code in a terminal: all tools, the
user's own settings, CLAUDE.md and MCP servers, and no permission prompts.

Every chat is also a Remote Control session, so it shows up in the Claude mobile app and on
claude.ai/code: each reply asks the CLI for the bridge (the SDK's `remote_control` control
request) and reattaches to the same remote session as the previous reply of that chat.

Every reply runs as a detached job that writes its stream-json to a file under JOBS, not
to a pipe: closing HyGra does not end it, and the next start follows the same file again.
"""
import ctypes
import glob
import json
import os
import shutil
import subprocess
import time

MODEL = "claude-opus-5-5"
SYSTEM = ("Je draait in HyGra, een desktop-chatapp van HydrarGyrum, met volledige toegang tot de computer "
          "van de gebruiker. Antwoord in de taal van de gebruiker (meestal Nederlands), helder en to the point. "
          "Gebruik Markdown (tabellen mogen); zet code altijd in een codeblok met de taal erbij. "
          "Wiskunde mag in LaTeX: $...$ inline en $$...$$ apart; de app rendert dat met KaTeX.")
WORKDIR = os.path.expanduser("~")   # starts in the home folder, like a terminal would
DATA = os.path.join(os.environ.get("LOCALAPPDATA", WORKDIR), "HyGra")
LIMITS = os.path.join(DATA, "limits.json")   # last known usage of the plan
JOBS = os.path.join(DATA, "jobs")            # one folder per reply: meta.json, in.jsonl, out.jsonl, err.txt
REMOTE = os.path.join(DATA, "remote.json")   # HyGra session id -> Remote Control session (cse_...)

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
DETACH = NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
BREAKAWAY = 0x01000000   # CREATE_BREAKAWAY_FROM_JOB: survive even if HyGra itself runs inside a job object


TOOL = "⁣tool "    # invisible markers: the page draws such a line as a tool call,
TIN = "⁣tin "      # its input, its output (matched by tool id) and a thinking summary;
TOUT = "⁣tout "    # all but the tool line itself only show with DETAILS on
THINK = "⁣think "
CLIP = 6000   # longest input value or output kept for the details


def _describe(inp):
    """A short summary of what a tool call does, for the line in the chat."""
    for k in ("command", "file_path", "path", "pattern", "url", "query", "description", "prompt"):
        if isinstance(inp.get(k), str) and inp[k].strip():
            v = " ".join(inp[k].split())
            return " · " + (v[:140] + "…" if len(v) > 140 else v)
    return ""


def _clip(v):
    if isinstance(v, str):
        return v if len(v) <= CLIP else v[:CLIP] + f"\n… ({len(v) - CLIP} tekens meer)"
    if isinstance(v, dict):
        return {k: _clip(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clip(x) for x in v[:50]]
    return v


def tool_lines(b, sub=False):
    """A tool_use block as the tool line plus its input; sub: called by a subagent."""
    inp = b.get("input") or {}
    detail = {"id": b.get("id"), "i": _clip(inp)}
    if sub:
        detail["sub"] = 1
    return f"{TOOL}{b.get('name')}{_describe(inp)}\n{TIN}{json.dumps(detail, ensure_ascii=False)}"


def result_line(b):
    """A tool_result block as one line: its text output, matched to the call by id."""
    c = b.get("content")
    if isinstance(c, list):
        c = "\n".join(x.get("text", "") if x.get("type") == "text" else "[afbeelding]" for x in c if isinstance(x, dict))
    out = {"id": b.get("tool_use_id"), "o": _clip(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))}
    if b.get("is_error"):
        out["err"] = 1
    return f"{TOUT}{json.dumps(out, ensure_ascii=False)}"


def think_line(text):
    return f"{THINK}{json.dumps(text.strip(), ensure_ascii=False)}"


def available():
    return shutil.which("claude") is not None


def _limits(info):
    """Keeps the plan's usage windows from a rate_limit_event (utilization 0..1, reset as unix time)."""
    win = info.get("unifiedWindows") or {}
    out = {k: {"used": w.get("utilization"), "resets": w.get("resetsAt")} for k, w in win.items()
           if k in ("five_hour", "seven_day")}
    if out:
        out["at"] = time.time()
        os.makedirs(os.path.dirname(LIMITS), exist_ok=True)
        with open(LIMITS, "w", encoding="utf-8") as f:
            json.dump(out, f)
    return out


def last_limits():
    try:
        with open(LIMITS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def refresh_limits():
    """A tiny Haiku call, just to read the current usage windows."""
    res = reply(None, True, "Antwoord met alleen: ok", lambda t: None, lambda s: None, model="claude-haiku-5-5")
    return res.get("limits") or last_limits()


def _remotes():
    try:
        with open(REMOTE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _remember_remote(session_id, bridge_id):
    r = _remotes()
    if bridge_id is None:
        r.pop(session_id, None)
    elif r.get(session_id) != bridge_id:
        r[session_id] = bridge_id
    else:
        return
    os.makedirs(DATA, exist_ok=True)
    with open(REMOTE, "w", encoding="utf-8") as f:
        json.dump(r, f)


def forget(session_id):
    """Deletes the Claude Code transcript of a HyGra chat."""
    for f in glob.glob(os.path.join(os.path.expanduser("~"), ".claude", "projects", "*", f"{session_id}.jsonl")):
        os.remove(f)
    _remember_remote(session_id, None)


def session_exists(session_id):
    """True once Claude Code has written a transcript for this session (also after a stopped first reply)."""
    return bool(glob.glob(os.path.join(os.path.expanduser("~"), ".claude", "projects", "*", f"{session_id}.jsonl")))


# ----- processes by pid (a job outlives the HyGra that started it, so there is no Popen to ask)
def _created(pid):
    """Creation time of a running process, or None if it is gone; with the pid it identifies the process."""
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x101000, False, pid)   # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        if k32.WaitForSingleObject(h, 0) != 0x102:   # not WAIT_TIMEOUT: it has exited
            return None
        t = [ctypes.c_ulonglong() for _ in range(4)]
        if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            return None
        return t[0].value
    finally:
        k32.CloseHandle(h)


def _alive(meta):
    return _created(meta["pid"]) == meta.get("created")


def stop(pid):
    """Ends a running reply, with everything Claude Code started under it (tools, shells)."""
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, creationflags=NO_WINDOW)


# ----- jobs
def _dir(job_id):
    return os.path.join(JOBS, "".join(c for c in job_id if c.isalnum() or c in "-_"))


def _meta(job_id):
    try:
        with open(os.path.join(_dir(job_id), "meta.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def start(job_id, session_id, first, prompt, model=MODEL, cwd=None):
    """Starts Claude Code detached for one reply; returns its pid, or an error dict."""
    exe = shutil.which("claude")
    if not exe:
        return {"error": "Claude Code (de 'claude'-opdracht) is niet gevonden. Installeer het en log in met je Pro-account."}
    cmd = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--include-partial-messages",
           "--model", model, "--append-system-prompt", SYSTEM, "--dangerously-skip-permissions",
           "--thinking-display", "summarized"]   # readable thinking summaries for DETAILS (hidden option)
    cmd += ["--no-session-persistence"] if session_id is None else ["--session-id" if first else "--resume", session_id]
    # never fall back to paid API billing: only the subscription login may be used
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    d = _dir(job_id)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    lines = []
    if session_id is not None:   # also visible on the phone: open the Remote Control bridge first
        rc = {"subtype": "remote_control", "enabled": True, "keep_session_on_exit": True}
        bridge = _remotes().get(session_id)
        if bridge:
            rc["reattach_session_id"] = bridge
        else:
            rc["name"] = "HyGra · " + (" ".join(prompt.split())[:60] or "chat")
        lines.append({"type": "control_request", "request_id": "rc", "request": rc})
    lines.append({"type": "user", "message": {"role": "user", "content": prompt}})
    with open(os.path.join(d, "in.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    with open(os.path.join(d, "in.jsonl"), "rb") as inp, open(os.path.join(d, "out.jsonl"), "wb") as out, \
            open(os.path.join(d, "err.txt"), "wb") as err:
        kw = dict(cwd=cwd if cwd and os.path.isdir(cwd) else WORKDIR, env=env, stdin=inp, stdout=out, stderr=err)
        try:
            proc = subprocess.Popen(cmd, creationflags=DETACH | BREAKAWAY, **kw)
        except OSError:   # the job object HyGra runs in does not allow breaking away
            proc = subprocess.Popen(cmd, creationflags=DETACH, **kw)
    meta = {"pid": proc.pid, "created": _created(proc.pid), "session": session_id, "started": time.time()}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return proc.pid


def _lines(job_id, meta):
    """Yields the stream-json lines of a job from the start, waiting for new ones until Claude Code has exited."""
    with open(os.path.join(_dir(job_id), "out.jsonl"), "rb") as f:
        part = b""
        while True:
            chunk = f.readline()
            if chunk:
                part += chunk
                if part.endswith(b"\n"):
                    yield part.decode("utf-8", "replace")
                    part = b""
                continue
            if not _alive(meta):   # exited: whatever is left in the file was written before that
                rest = part + f.read()
                for line in rest.decode("utf-8", "replace").splitlines():
                    yield line
                return
            time.sleep(0.1)


def follow(job_id, on_text, on_status):
    """Streams a job's answer through on_text (always from the beginning, so a reattached page gets all of it);
    returns the final text and token counts, or an error."""
    meta = _meta(job_id)
    if not meta:
        return {"error": "Dit antwoord is verloren gegaan (geen jobmap meer)."}
    result, text, limits, new_block, thought = None, [], None, False, None

    def emit(t):
        text.append(t)
        on_text(t)

    for line in _lines(job_id, meta):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "stream_event":
            ev = e["event"]
            if ev["type"] == "content_block_start":
                kind = ev["content_block"]["type"]
                on_status("denkt na" if kind == "thinking" else "werkt" if kind == "tool_use" else "schrijft")
                new_block = kind == "text"
                thought = [] if kind == "thinking" else None
            elif ev["type"] == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                if new_block and text and not text[-1].endswith("\n\n"):
                    emit("\n\n")   # a new text block after a tool call starts a new paragraph
                new_block = False
                emit(ev["delta"]["text"])
            elif ev["type"] == "content_block_delta" and ev["delta"].get("type") == "thinking_delta" and thought is not None:
                thought.append(ev["delta"].get("thinking", ""))
            elif ev["type"] == "content_block_stop" and thought is not None:
                if "".join(thought).strip():
                    emit(f"\n\n{think_line(''.join(thought))}\n\n")
                thought = None
        elif e.get("type") == "assistant":   # complete message: show each tool call as a line
            for b in e["message"].get("content", []):
                if b.get("type") == "tool_use":
                    emit(f"\n\n{tool_lines(b, bool(e.get('parent_tool_use_id')))}\n\n")
        elif e.get("type") == "user":   # tool output comes back as a user message
            c = (e.get("message") or {}).get("content")
            for b in c if isinstance(c, list) else []:
                if b.get("type") == "tool_result":
                    emit(f"\n{result_line(b)}\n\n")
        elif e.get("type") == "control_response" and meta.get("session"):
            r = e.get("response") or {}
            if r.get("subtype") == "success" and (r.get("response") or {}).get("bridge_session_id"):
                _remember_remote(meta["session"], r["response"]["bridge_session_id"])
        elif e.get("type") == "rate_limit_event":
            limits = _limits(e.get("rate_limit_info") or {})
        elif e.get("type") == "result":
            result = e
    if result is None:
        try:
            with open(os.path.join(_dir(job_id), "err.txt"), encoding="utf-8", errors="replace") as f:
                err = f.read().strip()
        except OSError:
            err = ""
        return {"error": err or "Claude Code stopte onverwacht.", "limits": limits}
    if result.get("is_error"):
        return {"error": result.get("result") or "Claude Code gaf een fout (gebruikslimiet bereikt?).", "limits": limits}
    u = result.get("usage") or {}
    tokens_in = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
    return {"text": "".join(text) or result.get("result") or "", "tokens_in": tokens_in,
            "tokens_out": u.get("output_tokens") or 0, "limits": limits}


def job_pid(job_id):
    meta = _meta(job_id)
    return meta["pid"] if meta and _alive(meta) else None


def done(job_id):
    """The page has stored the reply: the job folder can go."""
    shutil.rmtree(_dir(job_id), ignore_errors=True)


def tidy(days=7):
    """Removes job folders that no page will come back for: finished and older than a week."""
    for d in glob.glob(os.path.join(JOBS, "*")):
        meta = _meta(os.path.basename(d))
        started = meta.get("started", 0) if meta else os.path.getmtime(d)
        if time.time() - started > days * 86400 and not (meta and _alive(meta)):
            shutil.rmtree(d, ignore_errors=True)


def reply(session_id, first, prompt, on_text, on_status, model=MODEL, cwd=None, on_start=None, job_id=None):
    """Starts a job and follows it to the end. on_start(pid): the caller can stop it with stop(pid)."""
    own = job_id is None
    job_id = job_id or f"tmp{time.time_ns()}"
    pid = start(job_id, session_id, first, prompt, model, cwd)
    if isinstance(pid, dict):
        return pid
    if on_start:
        on_start(pid)
    try:
        return follow(job_id, on_text, on_status)
    finally:
        if own:
            done(job_id)
