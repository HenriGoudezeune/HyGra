"""Stage 1: train the multiscale residual VQ-VAE (text chunk <-> hierarchy of codes)."""
import argparse
import math
import os
import time
from dataclasses import asdict

import torch

from ncm.data import TokenData, get_encoder
from ncm.model import VQConfig, VQVAE
from ncm.thermal import cool_down

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--out", default="ckpt/tokenizer.pt")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--vq-weight", type=float, default=1.0)
p.add_argument("--eval-every", type=int, default=250)
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
cfg = VQConfig(vocab_size=get_encoder(args.data).vocab_size)
model = VQVAE(cfg).to(dev)
print(f"device={dev}  params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
data = TokenData(args.data)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda")


def lr_at(step, warmup=200):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    t = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(n=10):
    model.eval()
    accs, usage = [], torch.zeros(cfg.codebook_size, device=dev)
    for _ in range(n):
        x = data.batch("val", args.batch, cfg.context_length, dev)
        with torch.autocast(**amp):
            _, _, acc, idxs = model(x)
        accs.append(acc.item())
        usage += torch.bincount(idxs[-1].flatten(), minlength=cfg.codebook_size)
    model.train()
    prob = usage / usage.sum()
    ppl = torch.exp(-(prob[prob > 0] * prob[prob > 0].log()).sum()).item()
    return sum(accs) / len(accs), ppl


os.makedirs(os.path.dirname(args.out), exist_ok=True)
t0 = time.time()
for step in range(args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x = data.batch("train", args.batch, cfg.context_length, dev)
    with torch.autocast(**amp):
        ce, vq, acc, _ = model(x)
    loss = ce + args.vq_weight * vq
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        cool_down()
        print(f"step {step:5d}  ce {ce.item():.3f}  vq {vq.item():.4f}  acc {acc.item():.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        vacc, ppl = evaluate()
        print(f"  [val] token reconstruction acc {vacc:.3f}  finest-level code perplexity {ppl:.0f}", flush=True)
        torch.save({"cfg": asdict(cfg), "model": model.state_dict()}, args.out)
print(f"saved {args.out}")
