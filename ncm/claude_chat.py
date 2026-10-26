"""Claude Opus 5.5 as a chat partner in the app, through the local Claude Code CLI.

The CLI is logged in with the user's Claude subscription (Pro/Max), so chatting costs
nothing extra and counts toward the plan's normal usage limits; no API key is needed.
Each HyGra chat is one Claude Code session (`--session-id` on the first message,
`--resume` afterwards) with full access, like Claude Code in a terminal: all tools, the
user's own settings, CLAUDE.md and MCP servers, and no permission prompts.
"""
import glob
import json
import os
import shutil
import subprocess
import time

MODEL = "claude-opus-5-5"
SYSTEM = ("Je draait in HyGra, een desktop-chatapp van HydrarGyrum, met volledige toegang tot de computer "
          "van de gebruiker. Antwoord in de taal van de gebruiker (meestal Nederlands), helder en to the point. "
          "Gebruik Markdown; zet code altijd in een codeblok met de taal erbij.")
WORKDIR = os.path.expanduser("~")   # starts in the home folder, like a terminal would
LIMITS = os.path.join(os.environ.get("LOCALAPPDATA", WORKDIR), "HyGra", "limits.json")   # last known usage of the plan


TOOL = "\u2063tool "   # invisible marker: the page draws such a line as a tool call


def _describe(inp):
    """A short summary of what a tool call does, for the line in the chat."""
    for k in ("command", "file_path", "path", "pattern", "url", "query", "description", "prompt"):
        if isinstance(inp.get(k), str) and inp[k].strip():
            v = " ".join(inp[k].split())
            return " · " + (v[:140] + "…" if len(v) > 140 else v)
    return ""


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


def forget(session_id):
    """Deletes the Claude Code transcript of a HyGra chat."""
    for f in glob.glob(os.path.join(os.path.expanduser("~"), ".claude", "projects", "*", f"{session_id}.jsonl")):
        os.remove(f)


def reply(session_id, first, prompt, on_text, on_status, model=MODEL):
    """Streams the answer through on_text; returns the final text and token counts, or an error."""
    exe = shutil.which("claude")
    if not exe:
        return {"error": "Claude Code (de 'claude'-opdracht) is niet gevonden. Installeer het en log in met je Pro-account."}
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--model", model, "--append-system-prompt", SYSTEM, "--dangerously-skip-permissions"]
    cmd += ["--no-session-persistence"] if session_id is None else ["--session-id" if first else "--resume", session_id]
    # never fall back to paid API billing: only the subscription login may be used
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    proc = subprocess.Popen(cmd, cwd=WORKDIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    proc.stdin.write(prompt)
    proc.stdin.close()
    result, text, limits, new_block = None, [], None, False

    def emit(t):
        text.append(t)
        on_text(t)

    for line in proc.stdout:
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
            elif ev["type"] == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                if new_block and text and not text[-1].endswith("\n\n"):
                    emit("\n\n")   # a new text block after a tool call starts a new paragraph
                new_block = False
                emit(ev["delta"]["text"])
        elif e.get("type") == "assistant":   # complete message: show each tool call as a line
            for b in e["message"].get("content", []):
                if b.get("type") == "tool_use":
                    emit(f"\n\n{TOOL}{b['name']}{_describe(b.get('input') or {})}\n\n")
        elif e.get("type") == "rate_limit_event":
            limits = _limits(e.get("rate_limit_info") or {})
        elif e.get("type") == "result":
            result = e
    proc.wait()
    if result is None:
        err = proc.stderr.read().strip()
        return {"error": err or f"Claude Code stopte onverwacht (code {proc.returncode})."}
    if result.get("is_error"):
        return {"error": result.get("result") or "Claude Code gaf een fout (gebruikslimiet bereikt?).", "limits": limits}
    u = result.get("usage") or {}
    tokens_in = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
    return {"text": "".join(text) or result.get("result") or "", "tokens_in": tokens_in,
            "tokens_out": u.get("output_tokens") or 0, "limits": limits}
