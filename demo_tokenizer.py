"""Show how the VQ-VAE reconstructs a code snippet from coarse (1 code) to fine (all levels)."""
import argparse

import torch

from ncm.data import get_encoder
from ncm.model import VQConfig, VQVAE

SNIPPET = '''def binary_search(items, target):
    lo, hi = 0, len(items) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
'''

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", default="ckpt/tokenizer.pt")
p.add_argument("--file", help="snippet to reconstruct (default: built-in binary_search)")
p.add_argument("--data", default="data", help="directory holding bpe.json")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(args.tokenizer, map_location=dev)
cfg = VQConfig(**ck["cfg"])
vq = VQVAE(cfg).to(dev).eval()
vq.load_state_dict(ck["model"])
enc = get_encoder(args.data)
L, eot = cfg.context_length, enc.eot_token

text = open(args.file, encoding="utf-8").read() if args.file else SNIPPET
ids = enc.encode_ordinary(text)[:L]
ids = ids + [eot] * (L - len(ids))
x = torch.tensor([ids], device=dev)

with torch.no_grad():
    _, idxs, _, cums = vq.quant(vq.encode(x))
    n_codes = 0
    for k, cum in enumerate(cums):
        n_codes += cfg.levels[k]
        out = vq.decode(cum).argmax(-1)[0]
        acc = (out == x[0]).float().mean().item()
        rec = enc.decode([t for t in out.tolist() if t != eot])
        print(f"===== level {k}: {n_codes} codes so far, token accuracy {acc:.0%} =====")
        print(rec.rstrip(), "\n")
