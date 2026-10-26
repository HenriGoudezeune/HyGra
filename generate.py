"""Generate code with a trained NCM: chunk by chunk, each chunk coarse-to-fine."""
import argparse
import time

import torch

from eval_code import load, load_decoder, next_chunk
from ncm.data import get_encoder

p = argparse.ArgumentParser()
p.add_argument("--prompt", default="def fibonacci(n):\n")
p.add_argument("--prompt-file")
p.add_argument("--chunks", type=int, default=2, help="number of chunks (context_length tokens each)")
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--show-levels", action="store_true", help="print the text decoded at every scale")
p.add_argument("--tokenizer", default="ckpt/tokenizer.pt")
p.add_argument("--generator", default="ckpt/generator.pt")
p.add_argument("--data", default="data", help="directory holding bpe.json")
p.add_argument("--decoder", default="ckpt/decoder_ctx.pt", help="context decoder (used if present)")
p.add_argument("--seed", type=int)
args = p.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
vq, gen = load(args, dev)
decoder = load_decoder(args.decoder, vq, dev)
vcfg = vq.cfg

enc = get_encoder(args.data)
L, eot = vcfg.context_length, enc.eot_token
prompt = open(args.prompt_file, encoding="utf-8").read() if args.prompt_file else args.prompt
context = enc.encode_ordinary(prompt)
out = []

t0 = time.time()
with torch.no_grad():
    for c in range(args.chunks):
        new, cums = next_chunk(vq, gen, enc, context + out, args.temperature, args.top_k, dev, decoder)
        if args.show_levels:
            for k, cum in enumerate(cums):
                text = enc.decode(vq.decode(cum).argmax(-1)[0].tolist())
                print(f"--- chunk {c} level {k} ({vcfg.levels[k]} codes) ---\n{text}\n")
        if eot in new:
            out += new[:new.index(eot)]
            break
        out += new
dt = time.time() - t0

print(prompt + enc.decode(out))
print(f"\n[{len(out)} tokens in {dt:.2f}s, {len(vcfg.levels)} forward passes per {L}-token chunk]")
