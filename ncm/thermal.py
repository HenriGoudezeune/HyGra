"""Pause training while the GPU is too hot (laptop cooling)."""
import subprocess
import time


def gpu_temperature():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def cool_down(max_temp=82, resume_temp=72):
    t = gpu_temperature()
    if t is None or t < max_temp:
        return
    print(f"  GPU at {t}C, pausing until below {resume_temp}C", flush=True)
    while t is not None and t > resume_temp:
        time.sleep(10)
        t = gpu_temperature()
    print(f"  resumed at {t}C", flush=True)
