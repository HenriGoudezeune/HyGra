"""Time to produce one 128-token chunk: plain Qwen (token by token, KV cache) vs. the NCMs."""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_code import load, next_chunk
from ncm.data import get_encoder

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", default="ckpt/tokenizer.pt")
p.add_argument("--runs", type=int, default=5)
args = p.parse_args()
dev = "cuda"
prompt = 'def binary_search(items, target):\n    """Return the index of target in sorted items, or -1."""\n'
enc = get_encoder("data")


def timed(fn):
    fn()                                   # warm-up
    torch.cuda.synchronize(); t = time.time()
    for _ in range(args.runs):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / args.runs


results = {}
for name, path in [("NCM (34M, from scratch)", "ckpt/generator.pt"), ("Qwen-NCM (0.5B)", "ckpt/generator_qwen.pt")]:
    args.generator = path
    vq, gen = load(args, dev)
    ctx = enc.encode_ordinary(prompt)
    with torch.no_grad():
        results[name] = timed(lambda: next_chunk(vq, gen, enc, ctx, 0.0, 0, dev))
    del vq, gen; torch.cuda.empty_cache()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", dtype=torch.bfloat16).to(dev).eval()
ids = tok(prompt, return_tensors="pt").to(dev)
with torch.no_grad():
    results["Qwen 0.5B (token by token)"] = timed(lambda: qwen.generate(
        **ids, max_new_tokens=128, min_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id))
for name, t in results.items():
    print(f"{name:28s} {t * 1000:7.0f} ms per 128-token chunk")
