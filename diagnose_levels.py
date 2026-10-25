"""Per-level generator loss vs. unigram entropy of the codes (is each scale learned?).

Run it only while no training job uses the GPU: on a 4 GB card a second process spills
into shared system memory and slows the training down for good.
"""
import math

import torch
import torch.nn.functional as F

from ncm.data import TokenData
from ncm.model import GenConfig, NCMGenerator, VQConfig, VQVAE

dev = "cuda"
ck = torch.load("ckpt/tokenizer.pt", map_location=dev)
vcfg = VQConfig(**ck["cfg"])
vq = VQVAE(vcfg).to(dev).eval(); vq.load_state_dict(ck["model"])
gk = torch.load("ckpt/generator.pt", map_location=dev)
gen = NCMGenerator(GenConfig(**gk["cfg"]), vcfg).to(dev).eval(); gen.load_state_dict(gk["model"])
data = TokenData("data")
L, N = vcfg.context_length, gen.cfg.prefix_chunks
nl = len(vcfg.levels)
tot, acc = torch.zeros(nl), torch.zeros(nl)
counts = [torch.zeros(K, device=dev) for K in vcfg.codebook_sizes]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(8):
        x = data.batch("val", 16, (N + 1) * L, dev)
        pz = vq.encode_chunks(x[:, :N * L]).float()
        _, idxs, _, cums = vq.quant(vq.encode(x[:, N * L:]))
        li = [vq.quant.down(cums[k - 1], vcfg.levels[k]) for k in range(1, nl)]
        for k, (lg, t) in enumerate(zip(gen(pz, li), idxs)):
            tot[k] += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), t.reshape(-1)).item()
            acc[k] += (lg.argmax(-1) == t).float().mean().item()
    # unigram entropy of each level's codes = loss of a model that ignores the context
    for _ in range(256):
        _, idxs, _, _ = vq.quant(vq.encode(data.batch("val", 32, L, dev)))
        for c, t in zip(counts, idxs):
            c += torch.bincount(t.flatten(), minlength=c.numel())
for k, (l, c) in enumerate(zip(vcfg.levels, counts)):
    p = c[c > 0] / c.sum()
    h = -(p * p.log()).sum().item()
    print(f"level {k} ({l:2d} pos, {c.numel():4d} codes): loss {tot[k] / 8:.2f}  acc {acc[k] / 8:.2f}  "
          f"unigram entropy {h:.2f} (uniform {math.log(c.numel()):.2f})")
