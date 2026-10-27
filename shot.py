"""Screenshot of a HyGra window without bringing it to the front (for tests with app.py --background).

    python shot.py out.png [pid]

PrintWindow with PW_RENDERFULLCONTENT also captures a window that is covered by others, so the test
window can stay under Henri's windows. Without a pid it takes the lowest window titled "HyGra" in the
z-order (the --background one, not his own HyGra).
"""
import ctypes
import ctypes.wintypes as W
import sys

from PIL import Image

user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
user32.SetProcessDPIAware()


def find(pid=None):
    found = []

    @ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    def each(hwnd, _):
        title = ctypes.create_unicode_buffer(64)
        user32.GetWindowTextW(hwnd, title, 64)
        owner = W.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if title.value == "HyGra" and user32.IsWindowVisible(hwnd) and pid in (None, owner.value):
            found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    return found[-1] if pid is None and found else (found[0] if found else None)


def grab(hwnd):
    r = W.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    wdc = user32.GetWindowDC(hwnd)
    dc = gdi32.CreateCompatibleDC(wdc)
    bmp = gdi32.CreateCompatibleBitmap(wdc, w, h)
    gdi32.SelectObject(dc, bmp)
    user32.PrintWindow(hwnd, dc, 2)   # PW_RENDERFULLCONTENT: WebView2 / DirectComposition content too
    buf = ctypes.create_string_buffer(w * h * 4)
    head = (ctypes.c_uint32 * 10)(40, w, -h, 1 | 32 << 16, 0, 0, 0, 0, 0, 0)   # BITMAPINFOHEADER, top-down BGRA
    gdi32.GetDIBits(dc, bmp, 0, h, buf, head, 0)
    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(dc)
    user32.ReleaseDC(hwnd, wdc)
    return Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")


if __name__ == "__main__":
    hwnd = find(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    if not hwnd:
        sys.exit("no HyGra window")
    grab(hwnd).save(sys.argv[1])
    print(sys.argv[1])
