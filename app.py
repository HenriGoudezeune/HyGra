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

SIZE = (1280, 860)
LOG = os.path.join(os.environ.get("LOCALAPPDATA", ROOT), "HyGra", "hygra.log")
if sys.stderr is None:   # pythonw has no console: keep warnings and tracebacks (this run and the one before)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    try:
        os.replace(LOG, LOG[:-4] + ".prev.log")
    except OSError:
        pass
    sys.stderr = sys.stdout = open(LOG, "a", encoding="utf-8", buffering=1)
T0 = time.time()


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} +{time.time() - T0:5.1f}s  {msg}", file=sys.stderr, flush=True)


log("start")
# --background (for tests): no splash, the window never takes focus and opens under all other windows
BACKGROUND = "--background" in sys.argv
splash = None
if __name__ == "__main__" and sys.platform == "win32":
    # before anything heavy: own taskbar group + icon instead of Python's, and the loading screen at once
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("hydrargyrum.hygra")
    if not BACKGROUND:
        from ncm.splash import Splash
        splash = Splash(ROOT, *SIZE)

import bottle  # noqa: E402
import webview  # noqa: E402

from ncm import claude_chat, history  # noqa: E402

# pywebview sets no-cache on its files, but bottle's static_file overwrites those headers with its own, so
# WebView2 cached index.html heuristically and kept showing an old page after an update
_static_file = bottle.static_file


def _static_no_cache(*a, **kw):
    r = _static_file(*a, **kw)
    r.set_header("Cache-Control", "no-cache")   # still cheap: revalidates with ETag/Last-Modified (304)
    return r


bottle.static_file = _static_no_cache

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
        self._stopped = set()             # message ids the user stopped

    def _start_boot(self):
        """Called once the page (with its loading screen) is on screen: importing torch holds the GIL,
        so starting it earlier would only delay the window itself."""
        if not self._booting:
            self._booting = True
            threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self):
        global torch, cut_function, load, load_decoder, next_chunk
        claude_chat.spare()   # the first new chat should not wait for Claude Code to start either
        threading.Thread(target=claude_chat.tidy, daemon=True).start()
        try:
            log(f"cuda driver {'ready' if _cuda_init() else 'missing'}")
            import torch
            from eval_code import cut_function, load, load_decoder, next_chunk
            from ncm.data import get_encoder
            self.dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.enc = get_encoder("data")
            log(f"torch ready on {self.dev}")
        except Exception:
            import traceback
            traceback.print_exc()
            raise
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
        """Streams the reply into the page (hygra.delta / hygra.status) and returns the final turn.
        The chat's warm Claude Code process takes the message; it writes to a file, so if HyGra closes
        meanwhile, it finishes anyway and claude_attach picks it up."""
        pid = claude_chat.start(msg_id, session_id, bool(first), prompt, cwd=cwd)
        if isinstance(pid, dict):
            return pid
        return self.claude_attach(msg_id)

    def claude_steer(self, msg_id, text):
        """A message typed while that reply runs: Claude takes it in at its next step. None if the reply just ended."""
        return claude_chat.steer(msg_id, text)

    def claude_spare(self):
        """Session id for a new chat whose Claude Code process is already started."""
        return claude_chat.spare()

    def claude_warm(self, session_id, cwd=None):
        """The page shows this chat: get its Claude Code process ready for the next message."""
        threading.Thread(target=claude_chat.warm, args=(session_id, cwd), daemon=True).start()
        return True

    def _js(self, fn, *args):
        if self._window:
            self._window.evaluate_js(f"hygra.{fn}({', '.join(json.dumps(a) for a in args)})")

    def claude_attach(self, msg_id):
        """Follows a reply that is running (or finished) as a job, from the start of its output.
        Text goes to the page at most every 40 ms, and never sits in the buffer longer than that."""
        buf, lock, live = [], threading.Lock(), threading.Event()

        def flush():
            with lock:
                t = "".join(buf)
                buf.clear()
            if t:
                self._js("delta", msg_id, t)

        def pump():
            while not live.wait(0.04):
                flush()

        def on_text(t):
            with lock:
                buf.append(t)

        threading.Thread(target=pump, daemon=True).start()
        try:
            res = claude_chat.follow(msg_id, on_text, lambda s: self._js("status", msg_id, s),
                                     lambda late: self._js("late", msg_id, late))
        finally:
            live.set()
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
        if msg_id.startswith("m"):
            claude_chat.stop_job(msg_id)
        return True

    def claude_done(self, msg_id):
        """The page has saved the reply, so its job folder is no longer needed."""
        claude_chat.done(msg_id)
        return True

    def flash(self):
        """Lets the taskbar button blink until HyGra is brought to the front (only if it is not already)."""
        if self._hwnd and ctypes.windll.user32.GetForegroundWindow() != self._hwnd:
            _flash_taskbar(self._hwnd)

    def theme(self, name):
        """Remembers the page's theme outside WebView2, so the native splash (ncm/splash.py) starts in it too.
        Not in --background: a test instance has its own localStorage and must not change his splash."""
        if BACKGROUND:
            return True
        from ncm.splash import THEME_FILE
        try:
            with open(THEME_FILE, "w", encoding="utf-8") as f:
                f.write("cyber" if name == "cyber" else "hg")
        except OSError:
            pass
        return True

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


def _cuda_init():
    """Starts the NVIDIA driver (waking the laptop GPU) through ctypes, which lets go of the GIL meanwhile.
    torch.cuda does the same while holding it: 2 s with the GPU awake, far longer when it was asleep. The
    window's UI thread needs the GIL for every call from the page, so after 5 s Windows declared HyGra hung and
    swapped in a ghost window, which shows WebView2 as black. Afterwards torch finds the driver ready (0 s)."""
    try:
        return ctypes.WinDLL("nvcuda.dll").cuInit(0) == 0
    except OSError:   # no NVIDIA driver: torch stays on the CPU
        return False


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


def reveal(window, api):
    """The page has loaded in the hidden window: put it exactly where the splash is, show it underneath
    the splash, give WebView2 a few frames to paint, then drop the splash. Only then start the heavy
    boot (torch holds the GIL and would stall the swap)."""
    log("page loaded")
    if splash:
        time.sleep(0.1)   # first paint of the page (fonts, #boot)
        rect = splash.rect()
        splash.cover()
        if rect:
            ctypes.windll.user32.SetWindowPos(window.native.Handle.ToInt64(), None, *rect, 0x4 | 0x10)   # NOZORDER|NOACTIVATE
    if BACKGROUND:   # pywebview's show() also calls Activate(); show it behind everything instead
        hwnd = window.native.Handle.ToInt64()
        ctypes.windll.user32.ShowWindow(hwnd, 4)   # SW_SHOWNOACTIVATE
        ctypes.windll.user32.SetWindowPos(hwnd, 1, 0, 0, 0, 0, 0x1 | 0x2 | 0x10)   # HWND_BOTTOM, NOSIZE|NOMOVE|NOACTIVATE
    else:
        window.show()
    if splash:
        time.sleep(0.3)
        splash.close()
    log("window shown")
    api._start_boot()


def _window_bg():
    """The window's color before the page paints: the saved theme's ground, so nothing flashes between splash and page."""
    if sys.platform != "win32":
        return "#3B461C"
    from ncm.splash import saved_theme
    return "#101018" if saved_theme() == "cyber" else "#3B461C"


if __name__ == "__main__":
    api = Api()
    api._window = webview.create_window("HyGra", os.path.join(ROOT, "web", "index.html"), js_api=api,
                                       width=SIZE[0], height=SIZE[1], min_size=(420, 600),
                                       background_color=_window_bg(), hidden=splash is not None or BACKGROUND,
                                       focus=not BACKGROUND)
    if sys.platform == "win32":
        api._window.events.shown += lambda: set_icon(api._window)
        api._window.events.shown += lambda: setattr(api, "_hwnd", api._window.native.Handle.ToInt64() or None)
    revealed = threading.Event()   # loaded also fires on a reload; reveal only once
    api._window.events.loaded += lambda: revealed.is_set() or (revealed.set(), reveal(api._window, api))
    # not private: localStorage (funky mode, filters, ...) must survive a restart
    # a test instance gets its own port: its own origin, so it never shares localStorage or cache with his HyGra
    webview.start(private_mode=False, http_port=42002 if BACKGROUND else None,
                  storage_path=os.path.join(os.environ.get("LOCALAPPDATA", ROOT), "HyGra", "webview"))
    claude_chat.shutdown()
