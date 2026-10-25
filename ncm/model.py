"""Next-Concept Modeling (NCM), small re-implementation for code.

Stage 1: VQVAE maps a chunk of L BPE tokens to a T-length latent and quantizes it
with a multiscale residual VQ (levels coarse -> fine, VAR-style).
Stage 2: NCMGenerator predicts the codes of the next chunk scale by scale, conditioned
on the (continuous) encoding of the previous chunk. Generation of one chunk takes
len(levels) forward passes, each predicting all codes of a level in parallel.

Based on Kumbong, Khare, Re et al., "Next-Concept Modeling" (arXiv 2610.14162,
github.com/HazyResearch/ncm). Deviation: the decoder receives the quantized latent
through a straight-through estimator so the encoder trains on reconstruction loss.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQConfig:
    vocab_size: int = 50257
    context_length: int = 128          # BPE tokens per chunk
    d_model: int = 256
    n_head: int = 4
    enc_layers: int = 3
    dec_layers: int = 3
    latent_dim: int = 64
    levels: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    codebook_size: int = 1024
    beta: float = 0.25                 # commitment cost
    ema_decay: float = 0.99


@dataclass
class GenConfig:
    d_model: int = 384
    n_head: int = 6
    n_layers: int = 6
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, d, n_head, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, T, C)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.mlp(self.ln2(x)))


class MultiScaleRVQ(nn.Module):
    """Residual VQ over scales: each level quantizes the area-downsampled residual."""

    def __init__(self, levels, codebook_size, dim, beta=0.25, decay=0.99):
        super().__init__()
        self.levels, self.K, self.dim = list(levels), codebook_size, dim
        self.T = self.levels[-1]
        self.beta, self.decay = beta, decay
        n = len(self.levels)
        self.register_buffer("codebooks", torch.randn(n, codebook_size, dim) * 0.02)
        self.register_buffer("ema_count", torch.ones(n, codebook_size))
        self.register_buffer("ema_sum", self.codebooks.clone())
        self.register_buffer("initialized", torch.tensor(False))
        self.phi = nn.ModuleList(nn.Conv1d(dim, dim, 3, padding=1) for _ in self.levels)

    # --- helpers shared by training and generation ---
    def down(self, x, length):
        if x.size(1) == length:
            return x
        return F.interpolate(x.transpose(1, 2), size=length, mode="area").transpose(1, 2)

    def up(self, x):
        if x.size(1) == self.T:
            return x
        return F.interpolate(x.transpose(1, 2), size=self.T, mode="linear",
                             align_corners=False).transpose(1, 2)

    def level_contribution(self, k, idx):
        h = self.up(F.embedding(idx, self.codebooks[k]))
        return 0.5 * h + 0.5 * self.phi[k](h.transpose(1, 2)).transpose(1, 2)

    def nearest(self, k, r):
        cb = self.codebooks[k]
        d = r.pow(2).sum(-1, keepdim=True) - 2 * r @ cb.t() + cb.pow(2).sum(-1)
        return d.argmin(-1)

    @torch.no_grad()
    def _ema_update(self, k, r, idx):
        flat, idx = r.reshape(-1, self.dim), idx.reshape(-1)
        onehot = F.one_hot(idx, self.K).type_as(flat)
        self.ema_count[k].mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
        self.ema_sum[k].mul_(self.decay).add_(onehot.t() @ flat, alpha=1 - self.decay)
        # restart dead codes with random residual vectors from this batch
        dead = self.ema_count[k] < 1e-2
        if dead.any():
            pick = flat[torch.randint(0, flat.size(0), (int(dead.sum()),), device=flat.device)]
            self.ema_sum[k][dead] = pick
            self.ema_count[k][dead] = 1.0
        n = self.ema_count[k].sum()
        count = (self.ema_count[k] + 1e-5) / (n + self.K * 1e-5) * n
        self.codebooks[k].copy_(self.ema_sum[k] / count[:, None])

    @torch.no_grad()
    def _init_from(self, z):
        for k, l in enumerate(self.levels):
            flat = self.down(z, l).reshape(-1, self.dim)
            pick = flat[torch.randint(0, flat.size(0), (self.K,), device=z.device)]
            self.codebooks[k].copy_(pick + 0.01 * torch.randn_like(pick))
            self.ema_sum[k].copy_(self.codebooks[k])
        self.initialized.fill_(True)

    def forward(self, z):
        """z: (B, T, dim). Returns STE output, code indices per level, loss, cumulative f_hats."""
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            if self.training and not bool(self.initialized):
                self._init_from(z)
            f_rest, f_hat = z.detach().clone(), torch.zeros_like(z)
            idxs, cums, loss = [], [], 0.0
            for k, l in enumerate(self.levels):
                r = self.down(f_rest, l)
                idx = self.nearest(k, r)
                if self.training:
                    self._ema_update(k, r, idx)
                h = self.level_contribution(k, idx)
                f_hat = f_hat + h
                f_rest = f_rest - h.detach()
                loss = loss + F.mse_loss(f_hat, z.detach()) + self.beta * F.mse_loss(f_hat.detach(), z)
                idxs.append(idx)
                cums.append(f_hat.detach())
            loss = loss / len(self.levels)
            out = z + (f_hat - z).detach()
        return out, idxs, loss, cums


class VQVAE(nn.Module):
    def __init__(self, cfg: VQConfig):
        super().__init__()
        self.cfg = cfg
        L, d = cfg.context_length, cfg.d_model
        r = L // cfg.levels[-1]
        assert r * cfg.levels[-1] == L, "context_length must be a multiple of the finest level"
        self.wte = nn.Embedding(cfg.vocab_size, d)
        nn.init.normal_(self.wte.weight, std=0.02)
        self.enc_pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.enc = nn.ModuleList(Block(d, cfg.n_head) for _ in range(cfg.enc_layers))
        self.enc_ln = nn.LayerNorm(d)
        self.to_latent = nn.Conv1d(d, cfg.latent_dim, kernel_size=r, stride=r)
        self.quant = MultiScaleRVQ(cfg.levels, cfg.codebook_size, cfg.latent_dim, cfg.beta, cfg.ema_decay)
        self.from_latent = nn.ConvTranspose1d(cfg.latent_dim, d, kernel_size=r, stride=r)
        self.dec_pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.dec = nn.ModuleList(Block(d, cfg.n_head) for _ in range(cfg.dec_layers))
        self.dec_ln = nn.LayerNorm(d)

    def encode(self, ids):
        x = self.wte(ids) + self.enc_pos
        for b in self.enc:
            x = b(x)
        return self.to_latent(self.enc_ln(x).transpose(1, 2)).transpose(1, 2)

    def decode(self, f_hat):
        x = self.from_latent(f_hat.transpose(1, 2).to(self.dec_pos.dtype)).transpose(1, 2) + self.dec_pos
        for b in self.dec:
            x = b(x)
        return self.dec_ln(x) @ self.wte.weight.t()      # tied output embedding

    def forward(self, ids):
        f_hat, idxs, vq_loss, _ = self.quant(self.encode(ids))
        logits = self.decode(f_hat)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), ids.reshape(-1))
        acc = (logits.argmax(-1) == ids).float().mean()
        return ce, vq_loss, acc, idxs


class NCMGenerator(nn.Module):
    """Block-causal transformer over [prefix | level_0 | level_1 | ...]."""

    def __init__(self, cfg: GenConfig, vq: VQConfig):
        super().__init__()
        self.cfg, self.levels = cfg, list(vq.levels)
        P, d = vq.levels[-1], cfg.d_model
        self.prefix_len = P
        self.inp = nn.Linear(vq.latent_dim, d)
        self.sos = nn.Parameter(torch.randn(1, self.levels[0], d) * 0.02)
        S = P + sum(self.levels)
        self.pos = nn.Parameter(torch.randn(1, S, d) * 0.02)
        groups = torch.cat([torch.zeros(P, dtype=torch.long)] +
                           [torch.full((l,), k + 1, dtype=torch.long) for k, l in enumerate(self.levels)])
        self.register_buffer("group", groups, persistent=False)
        self.register_buffer("mask", groups[:, None] >= groups[None, :], persistent=False)
        self.lvl_emb = nn.Embedding(len(self.levels) + 1, d)
        self.blocks = nn.ModuleList(Block(d, cfg.n_head, cfg.dropout) for _ in range(cfg.n_layers))
        self.ln = nn.LayerNorm(d)
        self.heads = nn.ModuleList(nn.Linear(d, vq.codebook_size) for _ in self.levels)

    def forward(self, prefix_z, level_inputs):
        """prefix_z: (B, P, dim) encoding of the previous chunk.
        level_inputs: inputs for levels 1..n (each (B, l_k, dim)) = downsampled cumulative f_hat.
        Returns logits for levels 0..len(level_inputs)."""
        B = prefix_z.size(0)
        parts = [self.inp(prefix_z), self.sos.expand(B, -1, -1)] + [self.inp(li) for li in level_inputs]
        x = torch.cat(parts, dim=1)
        S = x.size(1)
        x = x + self.pos[:, :S] + self.lvl_emb(self.group[:S])
        mask = self.mask[:S, :S]
        for b in self.blocks:
            x = b(x, mask)
        x = self.ln(x)
        logits, start = [], self.prefix_len
        for k in range(1 + len(level_inputs)):
            l = self.levels[k]
            logits.append(self.heads[k](x[:, start:start + l]))
            start += l
        return logits

    def loss(self, prefix_z, quant: MultiScaleRVQ, idxs, cums):
        level_inputs = [quant.down(cums[k - 1], self.levels[k]) for k in range(1, len(self.levels))]
        logits = self.forward(prefix_z, level_inputs)
        lg = torch.cat(logits, dim=1).float()
        tgt = torch.cat(idxs, dim=1)
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1))
        acc = (lg.argmax(-1) == tgt).float().mean()
        return loss, acc

    @torch.no_grad()
    def generate(self, prefix_z, quant: MultiScaleRVQ, temperature=1.0, top_k=50):
        """Coarse-to-fine: one forward pass per level. Returns final f_hat and cumulative f_hats."""
        B = prefix_z.size(0)
        f_hat = torch.zeros(B, quant.T, quant.dim, device=prefix_z.device)
        level_inputs, cums = [], []
        for k in range(len(self.levels)):
            logits = self.forward(prefix_z, level_inputs)[k].float()
            if temperature <= 0:
                idx = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[..., -1:]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                idx = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(B, -1)
            f_hat = f_hat + quant.level_contribution(k, idx).float()
            cums.append(f_hat)
            if k + 1 < len(self.levels):
                level_inputs.append(quant.down(f_hat, self.levels[k + 1]))
        return f_hat, cums
