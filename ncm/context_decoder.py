"""VQ decoder that also reads the previous chunk of text.

The stage-1 decoder rebuilds each 128-token chunk from its codes alone, so identifiers that
were defined earlier (and are not in the chunk's codes precisely) come out wrong. This
decoder attends to the previous chunk's tokens as well. Encoder and codebooks stay frozen,
so the codes are unchanged and every trained generator keeps working with it.
"""
import copy

import torch
import torch.nn as nn

from .model import Block, VQVAE


class ContextDecoder(nn.Module):
    def __init__(self, vq: VQVAE, extra_layers=2):
        super().__init__()
        cfg = vq.cfg
        L, d = cfg.context_length, cfg.d_model
        self.cfg = cfg
        # start from the trained stage-1 decoder
        self.wte = copy.deepcopy(vq.wte)
        self.from_latent = copy.deepcopy(vq.from_latent)
        self.dec_pos = copy.deepcopy(vq.dec_pos)
        self.dec = nn.ModuleList([copy.deepcopy(b) for b in vq.dec] +
                                 [Block(d, cfg.n_head) for _ in range(extra_layers)])
        for b in self.dec[len(vq.dec):]:     # new blocks start as identity
            nn.init.zeros_(b.proj.weight); nn.init.zeros_(b.proj.bias)
            nn.init.zeros_(b.mlp[2].weight); nn.init.zeros_(b.mlp[2].bias)
        self.dec_ln = copy.deepcopy(vq.dec_ln)
        self.ctx_pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.ctx_type = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, f_hat, ctx_ids):
        """f_hat: (B, T, dim) quantized latent of the chunk; ctx_ids: (B, L) previous tokens."""
        L = self.cfg.context_length
        x = self.from_latent(f_hat.transpose(1, 2).to(self.dec_pos.dtype)).transpose(1, 2) + self.dec_pos
        c = self.wte(ctx_ids) + self.ctx_pos + self.ctx_type
        h = torch.cat([c, x], dim=1)
        for b in self.dec:
            h = b(h)
        return self.dec_ln(h[:, L:]) @ self.wte.weight.t()
