"""Per-level generator loss vs. unigram entropy of the codes (is each scale learned?)."""
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
tot = torch.zeros(len(vcfg.levels)); acc = torch.zeros(len(vcfg.levels)); usage = []
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(8):
        x = data.batch("val", 16, (N + 1) * L, dev)
        pz = vq.encode_chunks(x[:, :N * L]).float()
        _, idxs, _, cums = vq.quant(vq.encode(x[:, N * L:]))
        li = [vq.quant.down(cums[k - 1], vcfg.levels[k]) for k in range(1, len(vcfg.levels))]
        logits = gen(pz, li)
        for k, (lg, t) in enumerate(zip(logits, idxs)):
            tot[k] += F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), t.reshape(-1)).item()
            acc[k] += (lg.argmax(-1) == t).float().mean().item()
    # unigram entropy of each level's codes = loss of a model that ignores context
    x = data.batch("val", 256, L, dev)
    _, idxs, _, _ = vq.quant(vq.encode(x))
    for t in idxs:
        p = torch.bincount(t.flatten(), minlength=vcfg.codebook_size).float(); p = p[p > 0] / p.sum()
        usage.append(-(p * p.log()).sum().item())
for k, l in enumerate(vcfg.levels):
    print(f"level {k} ({l:2d} codes): loss {tot[k] / 8:.2f}  acc {acc[k] / 8:.2f}  unigram entropy {usage[k]:.2f}")

