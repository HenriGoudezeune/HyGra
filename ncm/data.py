"""Batch sampling from the uint16 token memmaps written by prepare.py."""
import os

import numpy as np
import torch

from ncm.bpe import CodeTokenizer


def get_encoder(data_dir="data"):
    return CodeTokenizer(os.path.join(data_dir, "bpe.json"))


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
