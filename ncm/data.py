"""Corpus preparation and batch sampling (GPT-2 BPE, uint16 memmaps)."""
import glob
import os

import numpy as np
import tiktoken
import torch


def get_encoder():
    return tiktoken.get_encoding("gpt2")


def build_corpus(src_dirs, out_dir, exts=(".py",), val_frac=0.02, seed=0):
    enc = get_encoder()
    files = []
    for d in src_dirs:
        for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if p.endswith(exts) and os.path.isfile(p) and "test" not in os.path.basename(p):
                files.append(p)
    files.sort()
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    n_val = max(1, int(len(files) * val_frac))
    splits = {"val": files[:n_val], "train": files[n_val:]}
    os.makedirs(out_dir, exist_ok=True)
    for name, fs in splits.items():
        ids = []
        for p in fs:
            try:
                with open(p, encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            ids.extend(enc.encode_ordinary(text))
            ids.append(enc.eot_token)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(os.path.join(out_dir, f"{name}.bin"))
        print(f"{name}: {len(fs)} files, {len(arr):,} tokens")


class TokenData:
    def __init__(self, data_dir):
        self.splits = {
            s: np.memmap(os.path.join(data_dir, f"{s}.bin"), dtype=np.uint16, mode="r")
            for s in ("train", "val")
        }

    def batch(self, split, batch_size, length, device):
        arr = self.splits[split]
        ix = np.random.randint(0, len(arr) - length - 1, size=batch_size)
        x = np.stack([arr[i:i + length].astype(np.int64) for i in ix])
        return torch.from_numpy(x).to(device, non_blocking=True)
