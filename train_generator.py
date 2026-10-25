"""Stage 2: train the next-concept generator on codes from the frozen VQ-VAE."""
import argparse
import math
import os
import time
from dataclasses import asdict

import torch

from ncm.data import TokenData, get_encoder
from ncm.model import GenConfig, NCMGenerator, VQConfig, VQVAE

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--tokenizer", default="ckpt/tokenizer.pt")
p.add_argument("--out", default="ckpt/generator.pt")
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=4e-4)
p.add_argument("--pad-prob", type=float, default=0.3,
               help="chance to left-pad the prefix, so short prompts work at inference")
p.add_argument("--eval-every", type=int, default=500)
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
ck = torch.load(args.tokenizer, map_location=dev)
vcfg = VQConfig(**ck["cfg"])
vq = VQVAE(vcfg).to(dev)
vq.load_state_dict(ck["model"])
vq.eval().requires_grad_(False)

gcfg = GenConfig()
gen = NCMGenerator(gcfg, vcfg).to(dev)
print(f"device={dev}  generator params={sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M")
data = TokenData(args.data)
eot = get_encoder().eot_token
L = vcfg.context_length
opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")


def lr_at(step, warmup=300):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    t = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def get_batch(split):
    x = data.batch(split, args.batch, 2 * L, dev)
    prefix, target = x[:, :L].clone(), x[:, L:]
    if split == "train" and args.pad_prob > 0:
        pad = torch.rand(args.batch, device=dev) < args.pad_prob
        cut = torch.randint(0, L, (args.batch,), device=dev)
        cols = torch.arange(L, device=dev)[None, :]
        prefix[pad[:, None] & (cols < cut[:, None])] = eot
    with torch.no_grad(), torch.autocast(**amp):
        prefix_z = vq.encode(prefix).float()
        _, idxs, _, cums = vq.quant(vq.encode(target))
    return prefix_z, idxs, cums


@torch.no_grad()
def evaluate(n=10):
    gen.eval()
    tot = [0.0, 0.0]
    for _ in range(n):
        pz, idxs, cums = get_batch("val")
        with torch.autocast(**amp):
            loss, acc = gen.loss(pz, vq.quant, idxs, cums)
        tot[0] += loss.item()
        tot[1] += acc.item()
    gen.train()
    return tot[0] / n, tot[1] / n


os.makedirs(os.path.dirname(args.out), exist_ok=True)
t0 = time.time()
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    pz, idxs, cums = get_batch("train")
    with torch.autocast(**amp):
        loss, acc = gen.loss(pz, vq.quant, idxs, cums)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"step {step:5d}  loss {loss.item():.3f}  code acc {acc.item():.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        vl, va = evaluate()
        print(f"  [val] loss {vl:.3f}  code acc {va:.3f}", flush=True)
        torch.save({"cfg": asdict(gcfg), "model": gen.state_dict()}, args.out)
print(f"saved {args.out}")
