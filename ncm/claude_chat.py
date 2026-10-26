"""Claude Opus 5.5 as a chat partner in the app, through the local Claude Code CLI.

The CLI is logged in with the user's Claude subscription (Pro/Max), so chatting costs
nothing extra and counts toward the plan's normal usage limits; no API key is needed.
Each HyGra chat is one Claude Code session (`--session-id` on the first message,
`--resume` afterwards), run with every tool switched off: a plain conversation.
"""
import json
import os
import shutil
import subprocess

MODEL = "claude-opus-5-5"
SYSTEM = ("Je bent Claude, de assistent in HyGra, een desktop-app van HydrarGyrum. "
          "Antwoord in de taal van de gebruiker (meestal Nederlands), helder en to the point. "
          "Gebruik Markdown; zet code altijd in een codeblok met de taal erbij.")
# a folder of its own, so no project CLAUDE.md or settings leak into the chats
WORKDIR = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "HyGra", "chat")


def available():
    return shutil.which("claude") is not None


def reply(session_id, first, prompt, on_text, on_status):
    """Streams the answer through on_text; returns the final text and token counts, or an error."""
    exe = shutil.which("claude")
    if not exe:
        return {"error": "Claude Code (de 'claude'-opdracht) is niet gevonden. Installeer het en log in met je Pro-account."}
    os.makedirs(WORKDIR, exist_ok=True)
    cmd = [exe, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--model", MODEL, "--tools", "", "--system-prompt", SYSTEM, "--disable-slash-commands",
           "--strict-mcp-config", "--setting-sources", "",
           "--session-id" if first else "--resume", session_id]
    # never fall back to paid API billing: only the subscription login may be used
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    proc = subprocess.Popen(cmd, cwd=WORKDIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    proc.stdin.write(prompt)
    proc.stdin.close()
    result, text = None, []
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
        elif e.get("type") == "result":
            result = e
    proc.wait()
    if result is None:
        err = proc.stderr.read().strip()
        return {"error": err or f"Claude Code stopte onverwacht (code {proc.returncode})."}
    if result.get("is_error"):
        return {"error": result.get("result") or "Claude Code gaf een fout (gebruikslimiet bereikt?)."}
    u = result.get("usage") or {}
    tokens_in = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
    return {"text": result.get("result") or "".join(text), "tokens_in": tokens_in,
            "tokens_out": u.get("output_tokens") or 0}
