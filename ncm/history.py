"""Earlier conversations from outside HyGra, for the "Eerder" list in the app.

Reads the transcripts that Claude Code keeps on this computer:
- Claude Code in a terminal and the Code tab of Claude Desktop: ~/.claude/projects/*/*.jsonl
- Cowork in Claude Desktop: one small Claude Code home per task under %APPDATA%/Claude.
Chats from the Chat tab of claude.ai / Claude Desktop live only in the cloud, so they are not here.
"""
import glob
import json
import os
import re
from datetime import datetime

from ncm.claude_chat import me_line, result_line, think_line, tool_lines

HOME = os.path.expanduser("~")
CODE = os.path.join(HOME, ".claude", "projects")
COWORK = os.path.join(os.environ.get("APPDATA", HOME), "Claude", "local-agent-mode-sessions")
INDEX = os.path.join(os.environ.get("LOCALAPPDATA", HOME), "HyGra", "history.json")   # titles per transcript, by mtime

_NOISE = ("<command-", "<local-command", "<system-reminder>", "Caveat:", "<task-notification>", "[Request interrupted")


def _ts(e):
    """An entry's timestamp as unix time (None if it has none), so tool steps keep their real durations."""
    try:
        return datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()
    except (KeyError, ValueError, AttributeError):
        return None


def _clean(text):
    """User text without the reminders and command wrappers Claude Code adds around it."""
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S).strip()
    return "" if text.startswith(_NOISE) else text


def _user_text(e):
    if e.get("isMeta") or e.get("isSidechain"):
        return ""
    c = e.get("message", {}).get("content")
    if isinstance(c, str):
        return _clean(c)
    parts = []
    for b in c or []:
        if b.get("type") == "text":
            parts.append(_clean(b.get("text", "")))
        elif b.get("type") == "image":
            parts.append("[afbeelding]")
    return "\n\n".join(p for p in parts if p)


def _files():
    """(path, source, cwd-hint, title-hint) for every main transcript (not subagents)."""
    out = [(f, "code", None, None) for f in glob.glob(os.path.join(CODE, "*", "*.jsonl"))]
    for meta in glob.glob(os.path.join(COWORK, "*", "*", "local_*.json")):
        try:
            with open(meta, encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        sid = m.get("cliSessionId")
        for f in glob.glob(os.path.join(meta[:-5], ".claude", "projects", "*", f"{sid}.jsonl")) if sid else []:
            out.append((f, "cowork", m.get("cwd"), m.get("title")))
    return out


def _scan(path):
    """Title, cwd, first prompt and number of user prompts, reading the file once."""
    title, cwd, first, n = None, None, None, 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"ai-title"' in line or '"type":"user"' in line or '"type": "user"' in line:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "ai-title":
                    title = e.get("aiTitle") or title
                elif e.get("type") == "user":
                    cwd = cwd or e.get("cwd")
                    t = _user_text(e)
                    if t:
                        n += 1
                        first = first or t
    return {"title": title, "cwd": cwd, "first": first, "prompts": n}


def sessions(skip=()):
    """All earlier conversations, newest first. skip: session ids HyGra already shows itself."""
    try:
        with open(INDEX, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, ValueError):
        index = {}
    fresh, out = {}, []
    for path, source, cwd, title in _files():
        sid = os.path.splitext(os.path.basename(path))[0]
        if sid in skip:
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        key = f"{st.st_mtime_ns}:{st.st_size}"
        info = index.get(path)
        if not info or info.get("key") != key:
            try:
                info = {"key": key, **_scan(path)}
            except OSError:
                continue
        fresh[path] = info
        if not info["prompts"]:
            continue   # only commands or an empty start, nothing to read back
        name = title or info["title"] or info["first"].split("\n")[0][:60]
        out.append({"path": path, "id": sid, "source": source, "title": name, "updated": st.st_mtime,
                    "cwd": info["cwd"] or cwd, "prompts": info["prompts"]})
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as fh:
        json.dump(fresh, fh)
    out.sort(key=lambda s: s["updated"], reverse=True)
    return out


def load(path):
    """The conversation as HyGra turns: a prompt with Claude's answer (text and tool lines)."""
    if not any(os.path.abspath(path).startswith(os.path.abspath(r)) for r in (CODE, COWORK)):
        raise ValueError("geen Claude Code-transcript")
    turns, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("isSidechain"):
                continue
            ts = _ts(e)
            if e.get("type") == "user":
                t = _user_text(e)
                if t:
                    cur = {"prompt": t, "model": "claude", "parts": []}
                    turns.append(cur)
            elif e.get("type") == "assistant" and cur is not None:
                for b in e.get("message", {}).get("content") or []:
                    if b.get("type") == "text" and b.get("text", "").strip():
                        cur["parts"].append(b["text"].strip())
                    elif b.get("type") == "tool_use":
                        cur["parts"].append(tool_lines(b, t=ts))
                    elif b.get("type") == "thinking" and b.get("thinking", "").strip():
                        cur["parts"].append(think_line(b["thinking"]))
            elif e.get("type") == "user" and cur is not None:   # tool output, for the details
                c = e.get("message", {}).get("content")
                for b in c if isinstance(c, list) else []:
                    if b.get("type") == "tool_result":
                        cur["parts"].append(result_line(b, t=ts))
            elif e.get("type") == "attachment" and cur is not None:   # a message sent while Claude was working
                at = e.get("attachment") or {}
                if at.get("type") == "queued_command" and isinstance(at.get("prompt"), str):
                    cur["parts"].append(me_line(at.get("source_uuid"), at["prompt"]))
    for t in turns:
        t["text"] = "\n\n".join(t.pop("parts")) or "(geen antwoord)"
    return turns
