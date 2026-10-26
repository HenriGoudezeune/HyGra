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
import sys
import threading
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import webview  # noqa: E402

from eval_code import cut_function, load, load_decoder, next_chunk  # noqa: E402
from ncm import claude_chat  # noqa: E402
from ncm.data import get_encoder  # noqa: E402

MODELS = {"qwen": "ckpt/generator_qwen.pt", "ncm": "ckpt/generator.pt"}


class Api:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc = get_encoder("data")
        self.lock = threading.Lock()
        self.m = {}
        self._window = None

    def _use(self, name):
        if self.m.get("name") == name:
            return
        self.m = {}
        torch.cuda.empty_cache()
        args = SimpleNamespace(tokenizer="ckpt/tokenizer.pt", generator=MODELS[name])
        vq, gen = load(args, self.dev)
        self.m = {"name": name, "vq": vq, "gen": gen, "dec": load_decoder("ckpt/decoder_ctx.pt", vq, self.dev)}

    def info(self):
        gpu = torch.cuda.get_device_name(0) if self.dev == "cuda" else "CPU"
        return {"device": gpu, "models": [k for k, v in MODELS.items() if os.path.exists(v)]}

    # ----- Claude (through Claude Code and the user's subscription)
    def has_claude(self):
        return claude_chat.available()

    def claude(self, session_id, first, prompt, msg_id):
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

        res = claude_chat.reply(session_id, bool(first), prompt, on_text, on_status)
        flush()
        return res

    # ----- local NCM models
    def warmup(self, model):
        with self.lock:
            self._use(model)
        return True

    @torch.no_grad()
    def complete(self, prompt, model, temperature, chunks):
        try:
            if model not in MODELS:
                raise ValueError("onbekend model")
            chunks = max(1, min(4, int(chunks)))
            with self.lock:
                self._use(model)
                vq, gen, dec, enc = self.m["vq"], self.m["gen"], self.m["dec"], self.enc
                context, out, levels = enc.encode_ordinary(prompt), [], []
                if self.dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for c in range(chunks):
                    def tick(k, c=c):   # live progress in the window: chunk c, level k done
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
        except Exception as e:  # shown in the chat instead of failing silently
            return {"error": str(e)}


if __name__ == "__main__":
    if sys.platform == "win32":   # own taskbar group + icon instead of Python's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("hydrargyrum.hygra")
    api = Api()
    api._window = webview.create_window("HyGra", os.path.join(ROOT, "web", "index.html"), js_api=api,
                                       width=1280, height=860, min_size=(420, 600), background_color="#3B461C")
    webview.start()
