"""Claude Opus 5.5 as a chat partner in the app, through the local Claude Code CLI.

The CLI is logged in with the user's Claude subscription (Pro/Max), so chatting costs
nothing extra and counts toward the plan's normal usage limits; no API key is needed.
Each HyGra chat is one Claude Code session (`--session-id` on the first message,
`--resume` afterwards), run with every tool switched off: a plain conversation.
"""
import glob
import json
import os
import shutil
import subprocess
import time

MODEL = "claude-opus-5-5"
SYSTEM = ("Je bent Claude, de assistent in HyGra, een desktop-app van HydrarGyrum. "
          "Antwoord in de taal van de gebruiker (meestal Nederlands), helder en to the point. "
          "Gebruik Markdown; zet code altijd in een codeblok met de taal erbij.")
# a folder of its own, so no project CLAUDE.md or settings leak into the chats
WORKDIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "HyGra", "chat")
LIMITS = os.path.join(WORKDIR, "..", "limits.json")   # last known 5-hour / weekly usage of the plan


def available():
    return shutil.which("claude") is not None


def _limits(info):
    """Keeps the plan's usage windows from a rate_limit_event (utilization 0..1, reset as unix time)."""
    win = info.get("unifiedWindows") or {}
    out = {k: {"used": w.get("utilization"), "resets": w.get("resetsAt")} for k, w in win.items()
           if k in ("five_hour", "seven_day")}
    if out:
        out["at"] = time.time()
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
    for f in glob.glob(os.path.join(os.path.expanduser("~"), ".claude", "projects", "*HyGra-chat", f"{session_id}.jsonl")):
        os.remove(f)


def reply(session_id, first, prompt, on_text, on_status, model=MODEL):
    """Streams the answer through on_text; returns the final text and token counts, or an error."""
    exe = shutil.which("claude")
    if not exe:
        return {"error": "Claude Code (de 'claude'-opdracht) is niet gevonden. Installeer het en log in met je Pro-account."}
    os.makedirs(WORKDIR, exist_ok=True)
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--model", model, "--tools", "", "--system-prompt", SYSTEM, "--disable-slash-commands",
           "--strict-mcp-config", "--setting-sources", ""]
    cmd += ["--no-session-persistence"] if session_id is None else ["--session-id" if first else "--resume", session_id]
    # never fall back to paid API billing: only the subscription login may be used
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    proc = subprocess.Popen(cmd, cwd=WORKDIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    proc.stdin.write(prompt)
    proc.stdin.close()
    result, text, limits = None, [], None
    for line in proc.stdout:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "stream_event":
            ev = e["event"]
            if ev["type"] == "content_block_start":
                on_status("denkt na" if ev["content_block"]["type"] == "thinking" else "schrijft")
            elif ev["type"] == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                text.append(ev["delta"]["text"])
                on_text(ev["delta"]["text"])
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
    return {"text": result.get("result") or "".join(text), "tokens_in": tokens_in,
            "tokens_out": u.get("output_tokens") or 0, "limits": limits}
