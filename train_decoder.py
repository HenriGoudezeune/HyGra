"""Stage 1b: train a decoder that reads the previous chunk too (encoder + codes frozen)."""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from ncm.context_decoder import ContextDecoder
from ncm.data import TokenData, get_encoder
from ncm.model import VQConfig, VQVAE
from ncm.thermal import cool_down

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--tokenizer", default="ckpt/tokenizer.pt")
p.add_argument("--out", default="ckpt/decoder_ctx.pt")
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--short-prob", type=float, default=0.3, help="chance to blank part of the context")
p.add_argument("--eval-every", type=int, default=500)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
ck = torch.load(args.tokenizer, map_location=dev)
vcfg = VQConfig(**ck["cfg"])
vq = VQVAE(vcfg).to(dev)
vq.load_state_dict(ck["model"])
vq.eval().requires_grad_(False)
dec = ContextDecoder(vq).to(dev)
opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
data = TokenData(args.data)
eot, L = get_encoder(args.data).eot_token, vcfg.context_length
amp = dict(device_type="cuda", dtype=torch.bfloat16)
start = 0
if args.resume and os.path.exists(args.out):
    k = torch.load(args.out, map_location=dev)
    dec.load_state_dict(k["model"]); opt.load_state_dict(k["opt"]); start = k["step"]
    print(f"resumed at step {start}", flush=True)


def lr_at(step, warmup=200):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))))


def get_batch(split):
    x = data.batch(split, args.batch, 2 * L, dev)
    ctx, tgt = x[:, :L].clone(), x[:, L:]
    if split == "train":
        blank = torch.rand(args.batch, device=dev) < args.short_prob
        cut = torch.randint(0, L + 1, (args.batch,), device=dev)
        ctx[blank[:, None] & (torch.arange(L, device=dev)[None] < cut[:, None])] = eot
    with torch.no_grad(), torch.autocast(**amp):
        f_hat, _, _, _ = vq.quant(vq.encode(tgt))
    return f_hat, ctx, tgt


def step_loss(f_hat, ctx, tgt):
    with torch.autocast(**amp):
        logits = dec(f_hat, ctx)
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1))
    return ce, (logits.argmax(-1) == tgt).float().mean()


@torch.no_grad()
def evaluate(n=10):
    dec.eval()
    acc_ctx = acc_base = 0.0
    for _ in range(n):
        f_hat, ctx, tgt = get_batch("val")
        acc_ctx += step_loss(f_hat, ctx, tgt)[1].item()
        with torch.autocast(**amp):
            acc_base += (vq.decode(f_hat).argmax(-1) == tgt).float().mean().item()
    dec.train()
    return acc_ctx / n, acc_base / n


os.makedirs(os.path.dirname(args.out), exist_ok=True)
if start == 0:
    a, b = evaluate()
    print(f"step     0  [val] acc with context {a:.4f}  stage-1 decoder {b:.4f}", flush=True)
t0 = time.time()
for step in range(start, args.steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    ce, acc = step_loss(*get_batch("train"))
    opt.zero_grad(set_to_none=True)
    ce.backward()
    torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        cool_down()
        print(f"step {step:5d}  ce {ce.item():.4f}  acc {acc.item():.4f}  {time.time() - t0:.0f}s", flush=True)
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        a, b = evaluate()
        print(f"  [val] acc with context {a:.4f}  stage-1 decoder {b:.4f}", flush=True)
        torch.save({"model": dec.state_dict(), "opt": opt.state_dict(), "step": step + 1}, args.out)
print(f"saved {args.out}")
