"""Draw the HyGra icon and add 'HyGra' to the Start menu and the desktop (Windows)."""
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
ICON = os.path.join(ROOT, "web", "icon.ico")
OLIVE, DEEP, ORANGE, MUSTARD, MINT, TEXT = "#3B461C", "#2E3714", "#FF7A1A", "#F2C14E", "#7FE0C8", "#F4E4BC"


def font(size):
    f = ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", size)
    try:
        f.set_variation_by_name("Bold")
    except (OSError, ValueError):
        pass
    return f


# the logo: mercury's periodic-table tile (80, Hg) with a cut corner and a faceted droplet
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
cut = 40
tile = [(cut, 6), (S - 6, 6), (S - 6, S - 6 - cut), (S - 6 - cut, S - 6), (6, S - 6), (6, cut)]
d.polygon(tile, fill=DEEP)
d.line(tile + [tile[0]], fill=ORANGE, width=20, joint="curve")
d.text((44, 26), "80", font=font(44), fill=MUSTARD)
d.polygon([(192, 34), (216, 66), (204, 86), (180, 86), (168, 66)], fill=MINT)
d.line([(192, 34), (192, 86)], fill=DEEP, width=6)
d.text((34, 96), "Hg", font=font(132), fill=TEXT)
img.save(ICON, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(os.path.join(ROOT, "web", "logo.png"))

desktop = subprocess.run(["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
                         capture_output=True, text=True).stdout.strip()
programs = os.path.join(os.environ["APPDATA"], r"Microsoft\Windows\Start Menu\Programs")
for folder in (programs, desktop):                      # remove the shortcuts of the earlier name
    old = os.path.join(folder, "NCM Chat.lnk")
    if os.path.exists(old):
        os.remove(old)

pythonw = os.path.join(ROOT, ".venv", "Scripts", "pythonw.exe")
for folder in (programs, desktop):
    lnk = os.path.join(folder, "HyGra.lnk")
    ps = (f"$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
          f"$s.TargetPath='{pythonw}';$s.Arguments='\"{os.path.join(ROOT, 'app.py')}\"';"
          f"$s.WorkingDirectory='{ROOT}';$s.IconLocation='{ICON}';$s.Description='HyGra (HydrarGyrum)';$s.Save()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print("shortcut:", lnk)
