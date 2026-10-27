"""The chat models' command-line agents, as the MODELLEN panel sees them: is it installed, who is logged in,
and a console window to install it, log in or log out. HyGra never handles passwords or tokens itself:
every login is the CLI's own (browser) flow, in a window the user sees.
"""
import base64
import json
import os
import shutil
import subprocess
import threading
import winreg

from ncm import other_chat
from ncm.claude_chat import NO_WINDOW

NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)

# per model: the CLI, and the commands behind the panel's buttons (with what the user has to do in the window)
CLI = {
    "claude": {
        "exe": "claude", "name": "Claude Code",
        "install": ('powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://claude.ai/install.ps1 | iex"',
                    "Claude Code wordt geinstalleerd."),
        "login": ("claude auth login", "Je browser opent: log in met je Claude-account - Pro of Max."),
        "logout": ("claude auth logout", "Je wordt uitgelogd bij Claude."),
    },
    "gpt": {
        "exe": "codex", "name": "Codex CLI",
        "install": ("npm install -g @openai/codex", "De Codex CLI wordt geinstalleerd via npm."),
        "login": ("codex login", "Je browser opent: log in met je ChatGPT-account - gratis kan ook."),
        "logout": ("codex logout", "Je wordt uitgelogd bij ChatGPT."),
    },
    "gemini": {
        "exe": "agy", "name": "Antigravity CLI",
        "install": ("winget install -e --id Google.AntigravityCLI --accept-source-agreements --accept-package-agreements",
                    "De Antigravity CLI wordt geinstalleerd via winget."),
        "login": ("agy", "Log in met je Google-account in de browser die opent. Ben je al ingelogd? Typ dan /logout en start opnieuw. Sluit dit venster met Ctrl+C als je klaar bent."),
        "logout": ("agy", "Typ /logout, bevestig, en sluit dit venster daarna met Ctrl+C."),
    },
    "local": {
        "exe": "ollama", "name": "Ollama",
        "install": ("winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements",
                    "Ollama wordt geinstalleerd - draait lokale modellen."),
        "login": (f"ollama pull {other_chat.LOCAL_MODEL}", f"{other_chat.LOCAL_NAME} wordt gedownload - ongeveer 19 GB."),
        "logout": (f"ollama rm {other_chat.LOCAL_MODEL}", f"{other_chat.LOCAL_NAME} wordt van je schijf verwijderd."),
    },
}
NODE = ("winget install -e --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements",
        "Node.js ontbreekt - nodig voor npm. Het wordt nu geinstalleerd. Klik daarna in HyGra nog eens op INSTALLEREN.")


def _refresh_path():
    """Something just got installed: take PATH fresh from the registry, as a new terminal would."""
    parts = []
    for root, key in ((winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, "Environment")):
        try:
            with winreg.OpenKey(root, key) as k:
                parts.append(os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]))
        except OSError:
            pass
    if parts:
        os.environ["PATH"] = ";".join(parts + [os.environ.get("PATH", "")])


def _run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, creationflags=NO_WINDOW)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def _claude():
    code, out = _run([shutil.which("claude"), "auth", "status"])
    try:
        s = json.loads(out[out.index("{"):])
    except ValueError:
        return {"ready": False, "detail": out.strip()[-200:]}
    if not s.get("loggedIn"):
        return {"ready": False}
    plan = (s.get("subscriptionType") or "").capitalize()
    return {"ready": True, "account": s.get("email"), "plan": plan,
            "warn": None if s.get("authMethod") == "claude.ai" else "geen abonnement: dit kan geld kosten"}


def _codex():
    code, out = _run([shutil.which("codex"), "login", "status"])
    if code != 0 or "Logged in" not in out:
        return {"ready": False}
    info = {"ready": True}
    try:   # the account and plan are in the id token Codex keeps (read only, never sent anywhere)
        with open(os.path.join(os.path.expanduser("~"), ".codex", "auth.json"), encoding="utf-8") as f:
            tok = json.load(f)["tokens"]["id_token"].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(tok + "=" * (-len(tok) % 4)))
        info["account"] = claims.get("email")
        info["plan"] = ((claims.get("https://api.openai.com/auth") or {}).get("chatgpt_plan_type") or "").capitalize()
    except (OSError, ValueError, KeyError, IndexError):
        pass
    if "API key" in out:
        info["warn"] = "ingelogd met een API-sleutel: dit kost geld"
    return info


def _gemini():
    rows = other_chat.gemini_rows(fresh=True)
    if not rows:
        return {"ready": False}
    best = other_chat.gemini_model()
    pro = any("pro" in r[0] for r in rows)
    return {"ready": True, "plan": "met Pro" if pro else "zonder Pro (enkel Flash)", "model_name": best[1]}


def _local():
    if not other_chat.local_ready():
        return {"ready": False, "detail": f"{other_chat.LOCAL_NAME} nog niet gedownload (19 GB)"}
    return {"ready": True, "plan": "op je laptop · gratis · niets verlaat je computer"}


def status():
    """{key: {installed, ready, account, plan, warn, label, name}} for the chat models, checked in parallel."""
    out = {}

    def one(key, fn):
        cli = CLI[key]
        found = other_chat.ollama_exe() if key == "local" else shutil.which(cli["exe"])
        s = {"installed": bool(found), "ready": False, "cli": cli["name"]}
        if s["installed"]:
            s.update(fn())
        out[key] = s
    threads = [threading.Thread(target=one, args=a) for a in (("claude", _claude), ("gpt", _codex), ("gemini", _gemini), ("local", _local))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for key, m in other_chat.models().items():   # labels of the model each account gets
        out[key].update(label=m["label"], name=m["name"])
    return out


def action(key, what, on_done):
    """Opens a console window that runs install / login / logout for that model; on_done() once it is closed."""
    cmd, note = CLI[key][what]
    if key == "local" and what != "install":   # ollama may not be on PATH yet right after installing it
        cmd = cmd.replace("ollama", f'"{other_chat.ollama_exe() or "ollama"}"', 1)
    if what == "install" and cmd.startswith("npm") and not shutil.which("npm"):
        cmd, note = NODE
    if what == "install" and cmd.startswith("winget") and not shutil.which("winget"):
        note = "winget ontbreekt: installeer App Installer uit de Microsoft Store en probeer opnieuw."
        cmd = "echo."
    title = f"HyGra - {CLI[key]['name']}"
    line = f'title {title} & echo {note} & echo. & {cmd} & echo. & echo Klaar: sluit dit venster om terug te gaan naar HyGra. & pause >nul'
    p = subprocess.Popen(f'cmd /c "{line}"', creationflags=NEW_CONSOLE)

    def wait():
        p.wait()
        _refresh_path()
        on_done()
    threading.Thread(target=wait, daemon=True).start()
    return True
