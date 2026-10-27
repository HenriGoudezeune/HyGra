"""HyGra (HydrarGyrum): desktop chat app (`uv run python app.py`, or the Start-menu
shortcut from install_shortcut.py).

A native window (pywebview, Edge WebView2) around web/index.html; the page calls Python
directly through `window.pywebview.api`. Two kinds of partner:
- Claude Opus 5.5 through Claude Code and the user's Claude subscription: a real chat.
- the local NCM models: code completers that write the next chunks coarse-to-fine.
  Only one of them sits on the GPU at a time (4 GB).
"""
import json
import ctypes
import os
import re
import sys
import threading
import time
import urllib.parse
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import webview  # noqa: E402

from ncm import claude_chat, history  # noqa: E402

# torch and the NCM code take ~4 s to import; Api._boot loads them while the window shows its loading screen
torch = cut_function = load = load_decoder = next_chunk = None

MODELS = {"qwen": "ckpt/generator_qwen.pt", "ncm": "ckpt/generator.pt"}


class Api:
    def __init__(self):
        self.dev = "cpu"
        self.enc = None
        self.lock = threading.Lock()
        self.m = {}
        self._window = None
        self._hwnd = None
        self._ready = threading.Event()
        self._booting = False
        self._procs = {}                  # message id -> running Claude Code process (one per chat)
        self._stopped = set()             # message ids the user stopped

    def _start_boot(self):
        """Called once the page (with its loading screen) is on screen: importing torch holds the GIL,
        so starting it earlier would only delay the window itself."""
        if not self._booting:
            self._booting = True
            threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        global torch, cut_function, load, load_decoder, next_chunk
        try:
            import torch
            from eval_code import cut_function, load, load_decoder, next_chunk
            from ncm.data import get_encoder
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.enc = get_encoder("data")
        finally:   # never leave the page waiting; a failure then shows up as an error in info()
            self._ready.set()

    def _use(self, name):
        self._ready.wait()
        if self.m.get("name") == name:
            return
        self.m = {}
        torch.cuda.empty_cache()
        args = SimpleNamespace(tokenizer="ckpt/tokenizer.pt", generator=MODELS[name])
        vq, gen = load(args, self.dev)
        self.m = {"name": name, "vq": vq, "gen": gen, "dec": load_decoder("ckpt/decoder_ctx.pt", vq, self.dev)}

    def info(self):
        self._ready.wait()   # the page keeps its loading screen up until this returns
        gpu = torch.cuda.get_device_name(0) if self.dev == "cuda" else "CPU"
        return {"device": gpu, "models": [k for k, v in MODELS.items() if os.path.exists(v)]}

    # ----- Claude (through Claude Code and the user's subscription)
    def has_claude(self):
        return claude_chat.available()

    def limits(self, refresh=False):
        return claude_chat.refresh_limits() if refresh else claude_chat.last_limits()

    def forget(self, session_id):
        claude_chat.forget(session_id)
        return True

    # ----- earlier conversations from Claude Code and Cowork (read from this computer)
    def history(self):
        return history.sessions()

    def open_history(self, path):
        try:
            return {"turns": history.load(path)}
        except (OSError, ValueError) as e:
            return {"error": str(e)}

    def claude(self, session_id, first, prompt, msg_id, cwd=None):
        """Streams the reply into the page (hygra.delta / hygra.status) and returns the final turn."""
        buf, last = [], [time.time()]

        def flush():
            if buf and self._window:
                self._window.evaluate_js(f"hygra.delta({json.dumps(msg_id)}, {json.dumps(''.join(buf))})")
                buf.clear()
            last[0] = time.time()

        def on_text(t):
            buf.append(t)
            if time.time() - last[0] > 0.05:
                flush()

        def on_status(s):
            if self._window:
                self._window.evaluate_js(f"hygra.status({json.dumps(msg_id)}, {json.dumps(s)})")

        res = claude_chat.reply(session_id, bool(first), prompt, on_text, on_status, cwd=cwd,
                                on_start=lambda p: self._procs.__setitem__(msg_id, p))
        self._procs.pop(msg_id, None)
        flush()
        if msg_id in self._stopped:
            self._stopped.discard(msg_id)
            return {"stopped": True, "limits": res.get("limits")}
        return res

    def has_session(self, session_id):
        return claude_chat.session_exists(session_id)

    def stop(self, msg_id):
        """Stop button: ends that Claude reply, or the local completion that is running."""
        self._stopped.add(msg_id)
        proc = self._procs.get(msg_id)
        if proc:
            claude_chat.stop(proc)
        return True

    def flash(self):
        """Lets the taskbar button blink until HyGra is brought to the front (only if it is not already)."""
        if self._hwnd and ctypes.windll.user32.GetForegroundWindow() != self._hwnd:
            _flash_taskbar(self._hwnd)

    def taskbar(self, busy):
        """Indeterminate progress on the taskbar button while something is thinking, so you see it from elsewhere."""
        if self._hwnd:
            _taskbar_progress(self._hwnd, 1 if busy else 0)   # TBPF_INDETERMINATE / TBPF_NOPROGRESS
        return True

    # ----- files named in a reply: links only for files that really exist
    def paths(self, items, cwd=None, bases=()):
        """Maps each mention (absolute, ~, file:///, with :line) to its full path, or None. A relative one is tried
        in the chat folder, then in the folders (and their parents) of files the chat already linked to."""
        return {s: _resolve(s, cwd, bases) for s in items}

    def open_path(self, path):
        try:
            os.startfile(path)   # default app; folders open in Explorer
            return True
        except OSError as e:
            return {"error": str(e)}

    # ----- local NCM models
    def warmup(self, model):
        with self.lock:
            self._use(model)
        return True

    def complete(self, prompt, model, temperature, chunks, msg_id=None):
        self._ready.wait()
        try:
            if model not in MODELS:
                raise ValueError("onbekend model")
            chunks = max(1, min(4, int(chunks)))
            with self.lock, torch.no_grad():
                self._use(model)
                vq, gen, dec, enc = self.m["vq"], self.m["gen"], self.m["dec"], self.enc
                context, out, levels = enc.encode_ordinary(prompt), [], []
                if self.dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for c in range(chunks):
                    def tick(k, c=c):   # live progress in the window: chunk c, level k done
                        if msg_id in self._stopped:
                            raise _Stopped
                        if self._window:
                            self._window.evaluate_js(f"progress({c}, {k}, {chunks})")
                    new, cums = next_chunk(vq, gen, enc, context + out, float(temperature), 20, self.dev, dec, tick)
                    levels = [enc.decode(vq.decode(c).argmax(-1)[0].tolist()) for c in cums]
                    if enc.eot_token in new:
                        out += new[:new.index(enc.eot_token)]
                        break
                    out += new
                if self.dev == "cuda":
                    torch.cuda.synchronize()
                ms = (time.time() - t0) * 1000
            text = enc.decode(out)
            if prompt.lstrip().startswith(("def ", "class ", "async def ")):
                text = cut_function(prompt, text)[len(prompt):]
            return {"completion": text, "levels": levels, "ms": round(ms), "chunks": chunks}
        except _Stopped:
            return {"stopped": True}
        except Exception as e:  # shown in the chat instead of failing silently
            return {"error": str(e)}
        finally:
            self._stopped.discard(msg_id)


class _Stopped(Exception):
    pass


def _flash_taskbar(hwnd):
    """FlashWindowEx: the taskbar button blinks orange until the window comes to the front."""
    class FLASHWINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("hwnd", ctypes.c_void_p), ("dwFlags", ctypes.c_uint),
                    ("uCount", ctypes.c_uint), ("dwTimeout", ctypes.c_uint)]
    info = FLASHWINFO(ctypes.sizeof(FLASHWINFO), hwnd, 0x3 | 0xC, 0, 0)   # FLASHW_ALL | FLASHW_TIMERNOFG
    ctypes.windll.user32.FlashWindowEx(ctypes.byref(info))


def _resolve(s, cwd=None, bases=()):
    p = urllib.parse.unquote(s.strip().strip("`'\"<>"))
    p = re.sub(r"^file:/*", "", p, flags=re.I)
    p = re.sub(r"(#L\d+.*|:\d+(:\d+)?(-\d+)?)$", "", p)   # file.py:12, file.py:12:3, file.py#L12
    if re.match(r"^[a-z]+://", p, re.I) or not p:
        return None
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return os.path.normpath(p) if os.path.exists(p) else None
    home = claude_chat.WORKDIR
    dirs = [cwd if cwd and os.path.isdir(cwd) else home]
    for b in bases:   # web/index.html after C:/.../ncm-code/app.py: try ncm-code, then up to (not past) home
        b = os.path.normpath(b)
        while b not in dirs and os.path.dirname(b) != b and os.path.normcase(b) != os.path.normcase(home):
            dirs.append(b)
            b = os.path.dirname(b)
    for d in dirs:
        q = os.path.normpath(os.path.join(d, p))
        if os.path.exists(q):
            return q
    return None


def _taskbar_progress(hwnd, state):
    """ITaskbarList3::SetProgressState through a raw COM vtable (pywin32 has no wrapper for it)."""
    ole32 = ctypes.windll.ole32
    ole32.CoInitialize(None)   # js_api calls arrive on their own thread
    clsid, iid = ctypes.create_string_buffer(16), ctypes.create_string_buffer(16)
    ole32.CLSIDFromString("{56FDF344-FD6D-11d0-958A-006097C9A090}", clsid)   # CLSID_TaskbarList
    ole32.IIDFromString("{EA1AFB91-9E28-4B86-90E9-9E9F8A5EEFAF}", iid)       # IID_ITaskbarList3
    obj = ctypes.c_void_p()
    if ole32.CoCreateInstance(clsid, None, 1, iid, ctypes.byref(obj)) or not obj:   # CLSCTX_INPROC_SERVER
        return
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    method = lambda i, *args: ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *args)(vtbl[i])
    method(3)(obj)                                                    # HrInit
    method(10, ctypes.c_void_p, ctypes.c_int)(obj, hwnd, state)       # SetProgressState
    method(2)(obj)                                                    # Release


def set_icon(window):
    """Gives the window (and so the taskbar button) the HyGra icon instead of Python's."""
    user32 = ctypes.windll.user32
    user32.LoadImageW.restype = ctypes.c_void_p
    user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    hwnd = window.native.Handle.ToInt64()   # our own form; another window may also be titled "HyGra"
    ico = os.path.join(ROOT, "web", "icon.ico")
    for which, size in ((1, 256), (0, 32)):   # ICON_BIG for the taskbar / Alt+Tab, ICON_SMALL for the title bar
        h = user32.LoadImageW(None, ico, 1, size, size, 0x10)   # IMAGE_ICON, LR_LOADFROMFILE
        if hwnd and h:
            user32.SendMessageW(hwnd, 0x80, which, h)   # WM_SETICON
    if hwnd:   # the taskbar takes its button icon from these window properties, not from WM_SETICON
        from win32com.propsys import propsys, pscon
        store = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
        for key, value in ((pscon.PKEY_AppUserModel_ID, "hydrargyrum.hygra"),
                           (pscon.PKEY_AppUserModel_RelaunchIconResource, ico + ",0"),
                           (pscon.PKEY_AppUserModel_RelaunchDisplayNameResource, "HyGra"),
                           (pscon.PKEY_AppUserModel_RelaunchCommand,
                            f'"{os.path.join(ROOT, ".venv", "Scripts", "pythonw.exe")}" "{os.path.join(ROOT, "app.py")}"')):
            store.SetValue(key, propsys.PROPVARIANTType(value))
        store.Commit()


if __name__ == "__main__":
    if sys.platform == "win32":   # own taskbar group + icon instead of Python's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("hydrargyrum.hygra")
    api = Api()
    api._window = webview.create_window("HyGra", os.path.join(ROOT, "web", "index.html"), js_api=api,
                                       width=1280, height=860, min_size=(420, 600), background_color="#3B461C")
    if sys.platform == "win32":
        api._window.events.shown += lambda: set_icon(api._window)
        api._window.events.shown += lambda: setattr(api, "_hwnd", api._window.native.Handle.ToInt64() or None)
    api._window.events.loaded += api._start_boot
    # not private: localStorage (funky mode, filters, ...) must survive a restart
    webview.start(private_mode=False,
                  storage_path=os.path.join(os.environ.get("LOCALAPPDATA", ROOT), "HyGra", "webview"))
