"""Claude Opus 5.5 as a chat partner in the app, through the local Claude Code CLI.

The CLI is logged in with the user's Claude subscription (Pro/Max), so chatting costs
nothing extra and counts toward the plan's normal usage limits; no API key is needed.
Each HyGra chat is one Claude Code session (`--session-id` on the first message,
`--resume` afterwards) with full access, like Claude Code in a terminal: all tools, the
user's own settings, CLAUDE.md and MCP servers, and no permission prompts.

Every chat is also a Remote Control session, so it shows up in the Claude mobile app and on
claude.ai/code: each reply asks the CLI for the bridge (the SDK's `remote_control` control
request) and reattaches to the same remote session as the previous reply of that chat.

Each chat keeps a warm Claude Code process (see Worker) that waits on stdin for the next message,
plus a spare one for the next new chat, so a message does not pay for Claude Code's startup. The
process writes its stream-json to a file under JOBS, not to a pipe: closing HyGra only ends its stdin,
so the reply it is on still finishes, and the next start follows the same file again. A reply is done
for the page as soon as Claude ends its turn, without waiting for the turn summary or the exit.
"""
import ctypes
import glob
import json
import os
import shutil
import subprocess
import threading
import time
import uuid

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
ME = "⁣me "        # a message you sent while the reply ran, where Claude took it in
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


def tool_lines(b, parent=None, t=None):
    """A tool_use block as the tool line plus its input; parent: the Agent call whose subagent made it (the page
    nests it there, so an agent team shows as a tree), t: when it started (unix time, for durations)."""
    inp = b.get("input") or {}
    detail = {"id": b.get("id"), "i": _clip(inp), "t": round(t or time.time(), 1)}
    if parent:
        detail["sub"] = 1
        detail["p"] = parent
    return f"{TOOL}{b.get('name')}{_describe(inp)}\n{TIN}{json.dumps(detail, ensure_ascii=False)}"


def result_line(b, t=None):
    """A tool_result block as one line: its text output, matched to the call by id; t: when it came back."""
    c = b.get("content")
    if isinstance(c, list):
        c = "\n".join(x.get("text", "") if x.get("type") == "text" else "[afbeelding]" for x in c if isinstance(x, dict))
    out = {"id": b.get("tool_use_id"), "o": _clip(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)),
           "t": round(t or time.time(), 1)}
    if b.get("is_error"):
        out["err"] = 1
    return f"{TOUT}{json.dumps(out, ensure_ascii=False)}"


def think_line(text):
    return f"{THINK}{json.dumps(text.strip(), ensure_ascii=False)}"


def me_line(uid, content):
    if isinstance(content, list):
        content = "\n".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return f"{ME}{json.dumps({'u': uid, 't': str(content or '').strip()}, ensure_ascii=False)}"


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


def _cmd(exe, model, session_id, first):
    cmd = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--include-partial-messages", "--replay-user-messages",   # the echo tells where a reply starts
           "--model", model, "--append-system-prompt", SYSTEM, "--dangerously-skip-permissions",
           "--thinking-display", "summarized"]   # readable thinking summaries for DETAILS (hidden option)
    return cmd + (["--no-session-persistence"] if session_id is None else ["--session-id" if first else "--resume", session_id])


def _env():
    # never fall back to paid API billing: only the subscription login may be used
    return {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}


def _cwd(cwd):
    return os.path.normcase(os.path.normpath(cwd if cwd and os.path.isdir(cwd) else WORKDIR))


# ----- warm workers: Claude Code takes 1-2 s to start (more to load a long chat) and the Remote Control bridge
# another half second, so each chat keeps its claude process waiting for the next message on stdin.
# Its output still goes to a file: when HyGra closes, stdin ends and Claude Code finishes the reply it is on.
MAX_WARM = 4           # warm processes at most (~300 MB each), the spare for a new chat included
IDLE = 20 * 60         # an unused one goes after this many seconds


class Worker:
    def __init__(self, exe, session_id, first, cwd=None, model=MODEL):
        self.id = f"w{time.time_ns()}"
        self.dir = _dir(self.id)
        self.out = os.path.join(self.dir, "out.jsonl")
        self.session, self.cwd, self.model = session_id, _cwd(cwd), model
        self.used, self.busy, self.rc = time.time(), None, False
        os.makedirs(self.dir, exist_ok=True)
        cmd = _cmd(exe, model, session_id, first)
        with open(self.out, "wb") as out, open(os.path.join(self.dir, "err.txt"), "wb") as err:
            kw = dict(cwd=self.cwd, env=_env(), stdin=subprocess.PIPE, stdout=out, stderr=err)
            try:
                self.proc = subprocess.Popen(cmd, creationflags=DETACH | BREAKAWAY, **kw)
            except OSError:   # the job object HyGra runs in does not allow breaking away
                self.proc = subprocess.Popen(cmd, creationflags=DETACH, **kw)
        self.created = _created(self.proc.pid)
        bridge = _remotes().get(session_id)
        if bridge:   # an existing chat: back on its Remote Control session right away
            self.remote(reattach_session_id=bridge)

    def send(self, obj):
        self.proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def remote(self, **how):
        self.send({"type": "control_request", "request_id": f"rc{time.time_ns()}",
                   "request": {"subtype": "remote_control", "enabled": True, "keep_session_on_exit": True, **how}})
        self.rc = True

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        """Ends stdin: Claude Code finishes what it is doing and exits by itself."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass


_pool = {}            # session id -> Worker
_spare = [None]       # a worker for the next new chat, its session id chosen in advance
_lock = threading.RLock()


def _evict():
    """Makes room for one more worker: closes the least recently used idle ones beyond MAX_WARM - 1."""
    live = [w for w in _pool.values() if w.alive()] + ([_spare[0]] if _spare[0] and _spare[0].alive() else [])
    idle = sorted((w for w in live if not w.busy and w is not _spare[0]), key=lambda w: w.used)
    for w in idle[:max(0, len(live) - MAX_WARM + 1)]:
        w.close()
        _pool.pop(w.session, None)


def _janitor():
    while True:
        time.sleep(60)
        with _lock:
            for sid, w in list(_pool.items()):
                if not w.alive():
                    _pool.pop(sid)
                elif not w.busy and time.time() - w.used > IDLE:
                    w.close()
                    _pool.pop(sid)


threading.Thread(target=_janitor, daemon=True).start()


def warm(session_id, cwd=None):
    """Starts (or keeps) a process for an existing chat, so its next message needs no startup."""
    exe = shutil.which("claude")
    if not exe or not session_exists(session_id):
        return False
    with _lock:
        w = _pool.get(session_id)
        if w and w.alive() and w.cwd == _cwd(cwd):
            w.used = time.time()
            return True
        if w and w.busy:
            return True
        if w:
            w.close()
        _evict()
        _pool[session_id] = Worker(exe, session_id, False, cwd)
    return True


def spare():
    """The session id for a new chat: a process for it is already running (or starts now)."""
    exe = shutil.which("claude")
    with _lock:
        w = _spare[0]
        if not (w and w.alive()):
            if not exe:
                return str(uuid.uuid4())
            _evict()
            w = _spare[0] = Worker(exe, str(uuid.uuid4()), True)
        return w.session


def _refill(delay=4):
    """A new spare once the reply that took the old one is under way (not to compete with its startup)."""
    def go():
        time.sleep(delay)
        spare()
    threading.Thread(target=go, daemon=True).start()


def _take(exe, session_id, first, cwd, model=MODEL):
    with _lock:
        w = _spare[0]
        if w and w.session == session_id and w.alive() and w.model == model:
            _spare[0] = None
            _refill()
        else:
            w = _pool.get(session_id)
            if not (w and w.alive() and not w.busy and w.cwd == _cwd(cwd) and w.model == model):
                if w:
                    w.close()
                _evict()
                w = Worker(exe, session_id, first and not session_exists(session_id), cwd, model)
        _pool[session_id] = w
        w.used = time.time()
        return w


def start(job_id, session_id, first, prompt, model=MODEL, cwd=None):
    """Hands one message to the chat's warm Claude Code process (starting one if needed); returns its pid, or an error dict."""
    exe = shutil.which("claude")
    if not exe:
        return {"error": "Claude Code (de 'claude'-opdracht) is niet gevonden. Installeer het en log in met je Pro-account."}
    d = _dir(job_id)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    uid = str(uuid.uuid4())
    for attempt in (0, 1):
        w = _take(exe, session_id, first, cwd, model)
        try:
            offset = os.path.getsize(w.out)
            if not w.rc:   # a new chat: open its Remote Control session, named after the first question
                w.remote(name="HyGra · " + (" ".join(prompt.split())[:60] or "chat"))
            w.send({"type": "user", "uuid": uid, "session_id": "", "parent_tool_use_id": None,
                    "message": {"role": "user", "content": prompt}})
            break
        except OSError:   # the warm process died in the meantime: once more with a fresh one
            with _lock:
                if _pool.get(session_id) is w:
                    _pool.pop(session_id)
            if attempt:
                return {"error": "Claude Code kon niet starten."}
    w.busy = job_id
    meta = {"pid": w.proc.pid, "created": w.created, "session": session_id, "started": time.time(),
            "out": w.out, "offset": offset, "uuid": uid}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return w.proc.pid


def _lines(path, meta, offset=0):
    """Yields the stream-json lines of a job from offset, waiting for new ones until Claude Code has exited."""
    with open(path, "rb") as f:
        f.seek(offset)
        part, idle = b"", 0
        while True:
            chunk = f.readline()
            if chunk:
                part += chunk
                idle = 0
                if part.endswith(b"\n"):
                    yield part.decode("utf-8", "replace")
                    part = b""
                continue
            if idle % 10 == 0 and not _alive(meta):   # exited: whatever is left was written before that
                rest = part + f.read()
                for line in rest.decode("utf-8", "replace").splitlines():
                    yield line
                return
            idle += 1
            time.sleep(0.02 if idle < 250 else 0.1)   # quick while text flows, calmer when Claude waits on a tool


def _job_worker(meta):
    with _lock:
        return next((w for w in _pool.values() if w.out == meta.get("out")), None)


def follow(job_id, on_text, on_status, on_late=None):
    """Streams a job's answer through on_text (always from the beginning, so a reattached page gets all of it);
    returns the final text and token counts, or an error. The reply counts as done as soon as Claude ends its
    turn; what Claude Code still sends after that (usage, the turn summary) goes to on_late in the background."""
    meta = _meta(job_id)
    if not meta:
        return {"error": "Dit antwoord is verloren gegaan (geen jobmap meer)."}
    out = meta.get("out") or os.path.join(_dir(job_id), "out.jsonl")   # older jobs: their own out.jsonl
    s = {"text": [], "limits": None, "new_block": False, "thought": None, "in": 0, "out": 0, "end": False,
         "started": not meta.get("uuid")}
    with _lock:   # messages sent during this reply that Claude Code has not taken in yet
        pending = _steers.setdefault(job_id, set(meta.get("steer") or []))

    def emit(t):
        s["text"].append(t)
        on_text(t)

    def handle(e):
        """One event; returns the result event that closes the turn."""
        typ = e.get("type")
        if typ == "control_response" and meta.get("session"):
            r = e.get("response") or {}
            if r.get("subtype") == "success" and (r.get("response") or {}).get("bridge_session_id"):
                _remember_remote(meta["session"], r["response"]["bridge_session_id"])
            return None
        if not s["started"]:   # the tail of an earlier reply of this process: skip up to our own message
            s["started"] = typ == "user" and e.get("uuid") == meta["uuid"]
            return None
        if typ == "user" and e.get("uuid") in pending:   # a message sent meanwhile: Claude takes it in here
            with _lock:
                pending.discard(e["uuid"])
            emit(f"\n\n{me_line(e['uuid'], (e.get('message') or {}).get('content'))}\n\n")
            on_status("werkt")
            return None
        if typ == "stream_event":
            ev = e["event"]
            main = not e.get("parent_tool_use_id")
            if ev["type"] == "message_start" and main:
                u = (ev.get("message") or {}).get("usage") or {}
                s["in"] = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
            elif ev["type"] == "message_delta" and main:
                s["out"] += (ev.get("usage") or {}).get("output_tokens") or 0
                if (ev.get("delta") or {}).get("stop_reason") not in (None, "tool_use", "pause_turn"):
                    s["end"] = True
            elif ev["type"] == "content_block_start":
                kind = ev["content_block"]["type"]
                on_status("denkt na" if kind == "thinking" else "werkt" if kind == "tool_use" else "schrijft")
                s["new_block"] = kind == "text"
                s["thought"] = [] if kind == "thinking" else None
            elif ev["type"] == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                if s["new_block"] and s["text"] and not s["text"][-1].endswith("\n\n"):
                    emit("\n\n")   # a new text block after a tool call starts a new paragraph
                s["new_block"] = False
                emit(ev["delta"]["text"])
            elif ev["type"] == "content_block_delta" and ev["delta"].get("type") == "thinking_delta" and s["thought"] is not None:
                s["thought"].append(ev["delta"].get("thinking", ""))
            elif ev["type"] == "content_block_stop" and s["thought"] is not None:
                if "".join(s["thought"]).strip():
                    emit(f"\n\n{think_line(''.join(s['thought']))}\n\n")
                s["thought"] = None
        elif typ == "assistant":   # complete message: show each tool call as a line
            for b in e["message"].get("content", []):
                if b.get("type") == "tool_use":
                    emit(f"\n\n{tool_lines(b, e.get('parent_tool_use_id'))}\n\n")
        elif typ == "user":   # tool output comes back as a user message
            c = (e.get("message") or {}).get("content")
            for b in c if isinstance(c, list) else []:
                if b.get("type") == "tool_result":
                    emit(f"\n{result_line(b)}\n\n")
        elif typ == "rate_limit_event":
            s["limits"] = _limits(e.get("rate_limit_info") or {})
        elif typ == "result":
            return e
        return None

    def events():
        for line in lines:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    def finished():
        with _lock:
            _steers.pop(job_id, None)
            w = _job_worker(meta)
            if w and w.busy == job_id:
                w.busy = None
                w.used = time.time()

    def over():
        """The turn ended; the reply too, unless a message sent during it still waits: its turn belongs to this reply."""
        with _lock:
            if pending:
                return False
            finished()   # from here on steer() refuses, so nothing new can slip in behind the end
            return True

    lines = _lines(out, meta, meta.get("offset", 0))
    result = None
    for e in events():
        result = handle(e)
        if (result or s["end"]) and over():
            break
        result, s["end"] = None, False
    if s["end"] and not result:   # done for the page; the rest of the turn is read on the side
        early = len(s["text"])

        def rest():
            for e in events():
                if handle(e):
                    break
            extra = "".join(s["text"][early:])
            if on_late and (s["limits"] or extra.strip()):
                on_late({"limits": s["limits"], "text": extra if extra.strip() else None})
        threading.Thread(target=rest, daemon=True).start()
        return {"text": "".join(s["text"]), "tokens_in": s["in"], "tokens_out": s["out"], "limits": s["limits"]}
    finished()
    if result is None:
        try:
            with open(os.path.join(os.path.dirname(out), "err.txt"), encoding="utf-8", errors="replace") as f:
                err = f.read().strip()[-4000:]
        except OSError:
            err = ""
        return {"error": err or "Claude Code stopte onverwacht.", "limits": s["limits"]}
    if result.get("is_error"):
        return {"error": result.get("result") or "Claude Code gaf een fout (gebruikslimiet bereikt?).", "limits": s["limits"]}
    u = result.get("usage") or {}
    tokens_in = s["in"] or sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    return {"text": "".join(s["text"]) or result.get("result") or "", "tokens_in": tokens_in,
            "tokens_out": s["out"] or u.get("output_tokens") or 0, "limits": s["limits"]}


_steers = {}   # job id -> uuids of messages sent during that reply that Claude Code has not taken in yet


def steer(job_id, text):
    """A message typed while the reply runs, as in Claude Code: it goes in at Claude's next step (after a tool
    call), or right after the reply as a turn of its own that still belongs to it. Returns its uuid, or None
    when the reply is already over (the page then sends it as a normal message)."""
    meta = _meta(job_id)
    with _lock:
        w = meta and _job_worker(meta)
        pending = _steers.get(job_id)
        if not w or w.busy != job_id or not w.alive() or pending is None:
            return None
        uid = str(uuid.uuid4())
        try:
            w.send({"type": "user", "uuid": uid, "session_id": "", "parent_tool_use_id": None,
                    "message": {"role": "user", "content": text}})
        except OSError:
            return None
        pending.add(uid)
        meta["steer"] = meta.get("steer", []) + [uid]   # a reattached page follows its turn too
        try:
            with open(os.path.join(_dir(job_id), "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except OSError:
            pass
    return uid


def job_pid(job_id):
    meta = _meta(job_id)
    return meta["pid"] if meta and _alive(meta) else None


def stop_job(job_id):
    """Stop button: interrupts the reply but keeps the chat's process warm; kills it if it does not listen."""
    meta = _meta(job_id)
    if not meta or not _alive(meta):
        return
    w = _job_worker(meta)
    if w and w.alive():
        try:   # messages sent meanwhile go back to the page, not to Claude: withdraw them first, then interrupt
            with _lock:
                _steers.get(job_id, set()).clear()
            for uid in meta.get("steer") or []:
                w.send({"type": "control_request", "request_id": f"cq{time.time_ns()}",
                        "request": {"subtype": "cancel_async_message", "message_uuid": uid}})
            w.send({"type": "control_request", "request_id": f"int{time.time_ns()}", "request": {"subtype": "interrupt"}})
        except OSError:
            w = None
    if not w:   # a reply from before a restart: no stdin to talk to
        stop(meta["pid"])
        return

    def check():
        time.sleep(4)
        if w.busy == job_id and w.alive():
            stop(w.proc.pid)
            with _lock:
                if _pool.get(w.session) is w:
                    _pool.pop(w.session)
    threading.Thread(target=check, daemon=True).start()


def done(job_id):
    """The page has stored the reply: the job folder can go, and the worker's output once nobody needs it."""
    meta = _meta(job_id)
    shutil.rmtree(_dir(job_id), ignore_errors=True)
    if meta and meta.get("out") and not _alive(meta):
        _drop_worker_dir(os.path.dirname(meta["out"]))


def _drop_worker_dir(wdir):
    for m in glob.glob(os.path.join(JOBS, "*", "meta.json")):
        try:
            with open(m, encoding="utf-8") as f:
                if os.path.normcase(os.path.dirname(json.load(f).get("out") or "")) == os.path.normcase(wdir):
                    return   # another reply still reads from it
        except (OSError, ValueError):
            continue
    shutil.rmtree(wdir, ignore_errors=True)


def tidy(days=7):
    """Removes job folders no page will come back for (finished and older than a week), and the output
    of warm processes from an earlier run once they have exited."""
    for d in glob.glob(os.path.join(JOBS, "*")):
        name = os.path.basename(d)
        if name.startswith("w"):
            continue
        meta = _meta(name)
        started = meta.get("started", 0) if meta else os.path.getmtime(d)
        if time.time() - started > days * 86400 and not (meta and _alive(meta)):
            shutil.rmtree(d, ignore_errors=True)
    with _lock:
        mine = {os.path.normcase(w.dir) for w in list(_pool.values()) + [_spare[0]] if w}
    for d in glob.glob(os.path.join(JOBS, "w*")):
        if os.path.normcase(d) not in mine and time.time() - os.path.getmtime(d) > 60:
            _drop_worker_dir(d)   # still open by a running claude: rmtree fails quietly, next time then


def shutdown():
    """HyGra closes: idle processes exit now, busy ones after their reply (their stdin ends either way)."""
    with _lock:
        for w in list(_pool.values()) + [_spare[0]]:
            if w:
                w.close()


def refresh_limits():
    """A tiny Haiku call, just to read the current usage windows (no tools or MCP servers, so it starts quickly)."""
    exe = shutil.which("claude")
    if not exe:
        return last_limits()
    cmd = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
           "--model", "claude-haiku-5-5", "--no-session-persistence", "--strict-mcp-config", "--tools", ""]
    p = subprocess.Popen(cmd, cwd=WORKDIR, env=_env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    msg = {"type": "user", "message": {"role": "user", "content": "Antwoord met alleen: ok"}}
    p.stdin.write((json.dumps(msg) + "\n").encode())
    p.stdin.close()
    found = None
    try:
        for line in p.stdout:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "rate_limit_event":
                found = _limits(e.get("rate_limit_info") or {})
                break
            if e.get("type") == "result":
                break
    finally:
        p.kill()
    return found or last_limits()
