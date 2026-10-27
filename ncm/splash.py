"""Native loading screen for HyGra (Windows only, plain ctypes: Win32 + GDI + GDI+).

pywebview and WebView2 need about a second before the page can paint, and until then the
window is empty or black. app.py therefore shows this window first, before it even imports
webview, and keeps the real window hidden. It is a pixel copy of the page's #boot screen,
so the swap (cover() -> show the real window underneath -> close()) cannot be seen.
"""
import ctypes
import math
import os
import threading
import time
import winreg
from ctypes import wintypes as W

user32, gdi32, kernel32 = ctypes.windll.user32, ctypes.windll.gdi32, ctypes.windll.kernel32
gdip = ctypes.windll.gdiplus
dwm = ctypes.windll.dwmapi

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, W.HWND, W.UINT, W.WPARAM, W.LPARAM)
HANDLE = ctypes.c_void_p

user32.DefWindowProcW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.CreateWindowExW.argtypes = [W.DWORD, W.LPCWSTR, W.LPCWSTR, W.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, W.HWND, HANDLE, HANDLE, HANDLE]
user32.CreateWindowExW.restype = W.HWND
user32.LoadImageW.argtypes = [HANDLE, W.LPCWSTR, W.UINT, ctypes.c_int, ctypes.c_int, W.UINT]
user32.LoadImageW.restype = HANDLE
user32.BeginPaint.restype = HANDLE
user32.SetWindowPos.argtypes = [W.HWND, W.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, W.UINT]
user32.PostMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
for f in ("CreateCompatibleDC", "CreateCompatibleBitmap", "SelectObject", "CreateSolidBrush", "CreateFontW"):
    getattr(gdi32, f).restype = HANDLE
gdi32.CreateCompatibleDC.argtypes = [HANDLE]
gdi32.CreateCompatibleBitmap.argtypes = [HANDLE, ctypes.c_int, ctypes.c_int]
gdi32.SelectObject.argtypes = [HANDLE, HANDLE]
gdi32.DeleteObject.argtypes = [HANDLE]
gdi32.DeleteDC.argtypes = [HANDLE]
gdi32.BitBlt.argtypes = [HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, HANDLE, ctypes.c_int,
                         ctypes.c_int, W.DWORD]
for f in ("SetTextColor", "SetBkMode", "SetTextCharacterExtra", "SetTextAlign"):
    getattr(gdi32, f).argtypes = [HANDLE, ctypes.c_int]
gdi32.TextOutW.argtypes = [HANDLE, ctypes.c_int, ctypes.c_int, W.LPCWSTR, ctypes.c_int]
gdi32.GetTextExtentPoint32W.argtypes = [HANDLE, W.LPCWSTR, ctypes.c_int, ctypes.POINTER(W.SIZE)]
gdi32.GetTextMetricsW.argtypes = [HANDLE, ctypes.c_void_p]
user32.FillRect.argtypes = [HANDLE, ctypes.POINTER(W.RECT), HANDLE]
gdip.GdipCreateFromHDC.argtypes = [HANDLE, ctypes.POINTER(HANDLE)]

OLIVE, OLIVE_DEEP = 0x3B461C, 0x2E3714
ORANGE, MUSTARD, MINT, TEXT = 0xFF7A1A, 0xF2C14E, 0x7FE0C8, 0xF4E4BC
WM_DESTROY, WM_CLOSE, WM_PAINT, WM_ERASEBKGND, WM_TIMER, WM_APP = 0x2, 0x10, 0xF, 0x14, 0x113, 0x8000
CLASS = "HyGraSplash"


def _bgr(rgb):   # GDI COLORREF
    return ((rgb & 0xFF) << 16) | (rgb & 0xFF00) | (rgb >> 16)


def _argb(rgb):  # GDI+ ARGB
    return 0xFF000000 | rgb


class _Paint(ctypes.Structure):
    _fields_ = [("hdc", HANDLE), ("erase", W.BOOL), ("rc", W.RECT), ("restore", W.BOOL), ("inc", W.BOOL),
                ("rgb", ctypes.c_byte * 32)]


class _WndClass(ctypes.Structure):
    _fields_ = [("cbSize", W.UINT), ("style", W.UINT), ("proc", WNDPROC), ("cls", ctypes.c_int), ("wnd", ctypes.c_int),
                ("inst", HANDLE), ("icon", HANDLE), ("cursor", HANDLE), ("bg", HANDLE), ("menu", W.LPCWSTR),
                ("name", W.LPCWSTR), ("iconsm", HANDLE)]


class _GdipStartup(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("cb", ctypes.c_void_p), ("nobg", W.BOOL), ("nocodecs", W.BOOL)]


class _TextMetric(ctypes.Structure):
    _fields_ = [("height", W.LONG), ("ascent", W.LONG), ("descent", W.LONG), ("rest", ctypes.c_byte * 48)]


class _PointF(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


def _dark_title():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


class Splash:
    def __init__(self, root, width, height, title="HyGra"):
        self.hwnd = None
        self._t0 = time.perf_counter()
        self._ready = threading.Event()
        fonts = os.path.join(root, "web", "vendor", "fonts")
        for f in ("ChakraPetch-Bold.ttf", "JetBrainsMono.ttf"):
            gdi32.AddFontResourceExW(os.path.join(fonts, f), 0x10, None)   # FR_PRIVATE: only for this process
        threading.Thread(target=self._run, args=(root, width, height, title), daemon=True).start()
        self._ready.wait(2)

    # ----- called from the app
    def rect(self):
        r = W.RECT()
        if not (self.hwnd and user32.GetWindowRect(self.hwnd, ctypes.byref(r))):
            return None
        return r.left, r.top, r.right - r.left, r.bottom - r.top

    def cover(self):
        """Stay on top while the real window appears underneath (its first frames may still be black)."""
        if self.hwnd:
            user32.SetWindowPos(self.hwnd, W.HWND(-1), 0, 0, 0, 0, 0x1 | 0x2 | 0x10)   # TOPMOST, NOSIZE|NOMOVE|NOACTIVATE

    def close(self):
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_APP, 0, 0)

    # ----- the window thread
    def _run(self, root, width, height, title):
        user32.SetProcessDPIAware()   # same as pywebview, so both windows use the same pixels
        self.s = s = user32.GetDpiForSystem() / 96
        token = ctypes.c_size_t()
        gdip.GdiplusStartup(ctypes.byref(token), ctypes.byref(_GdipStartup(1, None, False, False)), None)

        ico = os.path.join(root, "web", "icon.ico")
        self._proc = WNDPROC(self._wndproc)   # keep a reference, or ctypes frees the callback
        wc = _WndClass(ctypes.sizeof(_WndClass), 0x3, self._proc, 0, 0, kernel32.GetModuleHandleW(None),
                       user32.LoadImageW(None, ico, 1, 256, 256, 0x10), user32.LoadCursorW(None, 32512),
                       gdi32.CreateSolidBrush(_bgr(OLIVE)), None, CLASS,
                       user32.LoadImageW(None, ico, 1, 16, 16, 0x10))
        user32.RegisterClassExW(ctypes.byref(wc))

        # where pywebview will put its window: WinForms CenterScreen in the work area
        wa = W.RECT()
        user32.SystemParametersInfoW(0x30, 0, ctypes.byref(wa), 0)   # SPI_GETWORKAREA
        w = min(int(width * s), wa.right - wa.left)
        h = min(int(height * s), wa.bottom - wa.top)
        x, y = wa.left + (wa.right - wa.left - w) // 2, wa.top + (wa.bottom - wa.top - h) // 2
        self.hwnd = user32.CreateWindowExW(0, CLASS, title, 0x00CF0000, x, y, w, h, None, None,
                                           kernel32.GetModuleHandleW(None), None)   # WS_OVERLAPPEDWINDOW
        dark = ctypes.c_int(1 if _dark_title() else 0)
        backdrop = ctypes.c_int(2 if dark.value else 1)
        dwm.DwmSetWindowAttribute(self.hwnd, 20, ctypes.byref(dark), 4)       # same title bar as pywebview's
        dwm.DwmSetWindowAttribute(self.hwnd, 38, ctypes.byref(backdrop), 4)
        user32.ShowWindow(self.hwnd, 5)   # SW_SHOW
        user32.UpdateWindow(self.hwnd)
        user32.SetTimer(self.hwnd, 1, 16, None)
        self._ready.set()

        msg = W.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        gdip.GdiplusShutdown(token)

    def _wndproc(self, hwnd, msg, wp, lp):
        if msg == WM_ERASEBKGND:
            return 1
        if msg == WM_TIMER:
            user32.InvalidateRect(hwnd, None, False)
            return 0
        if msg == WM_PAINT:
            ps = _Paint()
            hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
            rc = W.RECT()
            user32.GetClientRect(hwnd, ctypes.byref(rc))
            mem = gdi32.CreateCompatibleDC(hdc)
            bmp = gdi32.CreateCompatibleBitmap(hdc, rc.right, rc.bottom)
            old = gdi32.SelectObject(mem, bmp)
            try:
                self._draw(mem, rc.right, rc.bottom)
            except Exception:   # a drawing bug must never take the app down; the plain olive stays
                pass
            gdi32.BitBlt(hdc, 0, 0, rc.right, rc.bottom, mem, 0, 0, 0x00CC0020)   # SRCCOPY
            gdi32.SelectObject(mem, old)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem)
            user32.EndPaint(hwnd, ctypes.byref(ps))
            return 0
        if msg == WM_CLOSE:   # the user closed the loading screen: stop starting up
            os._exit(0)
        if msg == WM_APP:
            user32.DestroyWindow(hwnd)
            return 0
        if msg == WM_DESTROY:
            self.hwnd = None
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wp, lp)

    # ----- drawing: the same layout as #boot in web/index.html, in CSS px * s
    def _draw(self, dc, W_, H_):
        s, t = self.s, time.perf_counter() - self._t0
        line = round(6 * s)
        self._fill(dc, 0, 0, W_, H_, ORANGE)
        self._fill(dc, line, line, W_ - 2 * line, H_ - 2 * line, OLIVE)

        msg_h = 14 * 1.32
        top = (H_ - (132 + 28 + 69 + 28 + 18 + 28 + msg_h) * s) / 2
        cx = W_ / 2

        g = HANDLE()
        gdip.GdipCreateFromHDC(dc, ctypes.byref(g))
        gdip.GdipSetSmoothingMode(g, 4)   # anti-alias
        k, ox, oy = 2.2 * s, cx - 66 * s, top   # svg viewBox 60 -> 132 px
        P = lambda pts, dx=0.0, dy=0.0: [(ox + (a + dx) * k, oy + (b + dy) * k) for a, b in pts]
        frame = P([(10, 3), (57, 3), (57, 50), (50, 57), (3, 57), (3, 10)])
        self._poly(g, frame, OLIVE_DEEP)
        self._poly(g, frame, ORANGE, pen=5 * k)
        bob = -4 * (1 - math.cos(2 * math.pi * t / 1.1)) / 2   # boot-drop: 0 -> -4 -> 0 every 1.1 s
        self._poly(g, P([(45, 9), (50, 16), (47, 20), (43, 20), (40, 16)], dy=bob), MINT)
        self._poly(g, P([(45, 9), (45, 20)], dy=bob), OLIVE_DEEP, pen=1.5 * k, closed=False)

        tx, ty, tw, th = cx - 150 * s, top + (132 + 28 + 69 + 28) * s, 300 * s, 18 * s
        c = 10 * s
        self._poly(g, [(tx + c, ty), (tx + tw, ty), (tx + tw, ty + th - c), (tx + tw - c, ty + th), (tx, ty + th),
                       (tx, ty + c)], OLIVE_DEEP)   # .cut-sm
        gdip.GdipSetClipRect(g, ctypes.c_float(tx), ctypes.c_float(ty), ctypes.c_float(tw), ctypes.c_float(th), 0)
        for frac, delay, color in ((0.34, 0, ORANGE), (0.14, -0.55, MUSTARD)):   # boot-slide, 1.1 s linear
            p = ((t - delay) % 1.1) / 1.1
            bx, bw = tx + (-0.36 + 1.36 * p) * tw, frac * tw
            self._poly(g, [(bx + c, ty), (bx + bw, ty), (bx + bw - c, ty + th), (bx, ty + th)], color)
        gdip.GdipDeleteGraphics(g)

        gdi32.SetBkMode(dc, 1)   # TRANSPARENT
        self._text(dc, "80", ox + 10 * k, oy + 17 * k, 10 * k, MUSTARD, baseline=True)
        self._text(dc, "Hg", ox + 9 * k, oy + 47 * k, 27 * k, TEXT, baseline=True)
        y = top + 160 * s
        self._centered(dc, [("H", ORANGE), ("Y", TEXT), ("G", ORANGE), ("RA", TEXT)], cx, y, 46 * s, 6 * s, 46 * s)
        self._centered(dc, [("HYDRARGYRUM", MUSTARD)], cx, y + 54 * s, 15 * s, 4 * s, 15 * s)
        self._centered(dc, [("OPSTARTEN…", MINT)], cx, ty + th + 28 * s, 14 * s, 2 * s, msg_h * s,
                       face="JetBrains Mono", weight=400)

    @staticmethod
    def _fill(dc, x, y, w, h, rgb):
        b = gdi32.CreateSolidBrush(_bgr(rgb))
        user32.FillRect(dc, ctypes.byref(W.RECT(int(x), int(y), int(x + w), int(y + h))), b)
        gdi32.DeleteObject(b)

    @staticmethod
    def _poly(g, pts, rgb, pen=None, closed=True):
        arr = (_PointF * len(pts))(*[_PointF(a, b) for a, b in pts])
        obj = HANDLE()
        if pen is None:
            gdip.GdipCreateSolidFill(_argb(rgb), ctypes.byref(obj))
            gdip.GdipFillPolygon(g, obj, arr, len(pts), 0)
            gdip.GdipDeleteBrush(obj)
        else:
            gdip.GdipCreatePen1(_argb(rgb), ctypes.c_float(pen), 2, ctypes.byref(obj))   # UnitPixel
            gdip.GdipSetPenLineJoin(obj, 0)   # miter, like the svg
            (gdip.GdipDrawPolygon if closed else gdip.GdipDrawLines)(g, obj, arr, len(pts))
            gdip.GdipDeletePen(obj)

    @staticmethod
    def _font(px, face="Chakra Petch", weight=700):
        return gdi32.CreateFontW(-round(px), 0, 0, 0, weight, 0, 0, 0, 1, 0, 0, 5, 0, face)   # CLEARTYPE_QUALITY

    def _text(self, dc, s_, x, y, px, rgb, baseline=False):
        f = self._font(px)
        old = gdi32.SelectObject(dc, f)
        gdi32.SetTextAlign(dc, 24 if baseline else 0)   # TA_BASELINE
        gdi32.SetTextColor(dc, _bgr(rgb))
        gdi32.TextOutW(dc, round(x), round(y), s_, len(s_))
        gdi32.SetTextAlign(dc, 0)
        gdi32.SelectObject(dc, old)
        gdi32.DeleteObject(f)

    def _centered(self, dc, parts, cx, y, px, spacing, line_h, face="Chakra Petch", weight=700):
        """One line of text with CSS letter-spacing (also after the last letter), centered, line box from y."""
        f = self._font(px, face, weight)
        old = gdi32.SelectObject(dc, f)
        gdi32.SetTextCharacterExtra(dc, round(spacing))
        tm = _TextMetric()
        gdi32.GetTextMetricsW(dc, ctypes.byref(tm))
        sizes = []
        for text, _ in parts:
            sz = W.SIZE()
            gdi32.GetTextExtentPoint32W(dc, text, len(text), ctypes.byref(sz))
            sizes.append(sz.cx)
        x = cx - sum(sizes) / 2
        ty = y + (line_h - tm.ascent - tm.descent) / 2
        for (text, rgb), w in zip(parts, sizes):
            gdi32.SetTextColor(dc, _bgr(rgb))
            gdi32.TextOutW(dc, round(x), round(ty), text, len(text))
            x += w
        gdi32.SetTextCharacterExtra(dc, 0)
        gdi32.SelectObject(dc, old)
        gdi32.DeleteObject(f)
