"""Stage 2 on a pretrained backbone: Qwen2.5-Coder reads the text so far and predicts the
next chunk's concept codes coarse-to-fine (frozen VQ-VAE from stage 1)."""
import argparse
import math
import os
import random
import time
from dataclasses import asdict

import torch

from ncm.data import TokenData, get_encoder
from ncm.model import VQConfig, VQVAE
from ncm.qwen_gen import QwenConceptGenerator, QwenGenConfig
from ncm.thermal import cool_down

p = argparse.ArgumentParser()
p.add_argument("--data", default="data")
p.add_argument("--tokenizer", default="ckpt/tokenizer.pt")
p.add_argument("--out", default="ckpt/generator_qwen.pt")
p.add_argument("--steps", type=int, default=12000)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
p.add_argument("--lr-new", type=float, default=5e-4, help="slot inputs and heads")
p.add_argument("--prefix-tokens", type=int, default=384, help="context, in stage-1 BPE tokens")
p.add_argument("--short-prob", type=float, default=0.3, help="chance of a short prefix (short prompts)")
p.add_argument("--eval-every", type=int, default=250)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
ck = torch.load(args.tokenizer, map_location=dev)
vcfg = VQConfig(**ck["cfg"])
vq = VQVAE(vcfg).to(dev)
vq.load_state_dict(ck["model"])
vq.eval().requires_grad_(False)

gcfg = QwenGenConfig()
gen = QwenConceptGenerator(gcfg, vcfg).to(dev)
gen.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
gen.backbone.enable_input_require_grads()
lora = [q for n, q in gen.named_parameters() if q.requires_grad and n.startswith("backbone.")]
new = [q for n, q in gen.named_parameters() if q.requires_grad and not n.startswith("backbone.")]
print(f"trainable: LoRA {sum(q.numel() for q in lora) / 1e6:.1f}M, new {sum(q.numel() for q in new) / 1e6:.1f}M",
      flush=True)
opt = torch.optim.AdamW([{"params": lora, "lr": args.lr, "weight_decay": 0.0},
                         {"params": new, "lr": args.lr_new, "weight_decay": 0.01}], betas=(0.9, 0.95))
base_lr = [g["lr"] for g in opt.param_groups]
data = TokenData(args.data)
enc = get_encoder(args.data)
eot, L = enc.eot_token, vcfg.context_length
amp = dict(device_type="cuda", dtype=torch.bfloat16)

start = 0
if args.resume and os.path.exists(args.out):
    gk = torch.load(args.out, map_location=dev)
    gen.load_state_dict(gk["model"], strict=False)
    opt.load_state_dict(gk["opt"])
    start = gk["step"]
    print(f"resumed from {args.out} at step {start}", flush=True)


def lr_scale(step, warmup=200):
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, args.steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))


def prefix_text(ids, train):
    """Context = tokens after the last document break; sometimes cut short, like a prompt."""
    if eot in ids:
        ids = ids[len(ids) - ids[::-1].index(eot):]
    if train and random.random() < args.short_prob:
        ids = ids[len(ids) - random.randint(0, len(ids)):]
    return enc.decode(ids)


def get_batch(split):
    x = data.batch(split, args.batch, args.prefix_tokens + L, dev)
    texts = [prefix_text(row, split == "train") for row in x[:, :args.prefix_tokens].tolist()]
    prefix = gen.encode_prefix(texts, dev)
    with torch.no_grad(), torch.autocast(**amp):
        _, idxs, _, cums = vq.quant(vq.encode(x[:, args.prefix_tokens:]))
    return prefix, idxs, cums


@torch.no_grad()
def evaluate(n=10):
    gen.eval()
    tot = [0.0, 0.0]
    for _ in range(n):
        prefix, idxs, cums = get_batch("val")
        with torch.autocast(**amp):
            loss, acc = gen.loss(prefix, vq.quant, idxs, cums)
        tot[0] += loss.item()
        tot[1] += acc.item()
    gen.train()
    return tot[0] / n, tot[1] / n


os.makedirs(os.path.dirname(args.out), exist_ok=True)
gen.train()
t0 = time.time()
for step in range(start, args.steps):
    for g, lr in zip(opt.param_groups, base_lr):
        g["lr"] = lr * lr_scale(step)
    prefix, idxs, cums = get_batch("train")
    with torch.autocast(**amp):
        loss, acc = gen.loss(prefix, vq.quant, idxs, cums)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lora + new, 1.0)
    opt.step()
    if step % 50 == 0:
        cool_down()
        print(f"step {step:5d}  loss {loss.item():.3f}  code acc {acc.item():.3f}  "
              f"{time.time() - t0:.0f}s  {torch.cuda.max_memory_reserved() / 2**30:.2f}GB", flush=True)
    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
        vl, va = evaluate()
        print(f"  [val] loss {vl:.3f}  code acc {va:.3f}", flush=True)
        torch.save({"kind": "qwen", "cfg": asdict(gcfg), "model": gen.trainable_state(),
                    "opt": opt.state_dict(), "step": step + 1}, args.out)
print(f"saved {args.out}")
