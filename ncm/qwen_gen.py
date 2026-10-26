"""Next-concept generator on a pretrained Qwen2.5-Coder backbone.

The text so far goes in as ordinary Qwen tokens; after it come the concept slots of the
next chunk ([level 0 | level 1 | ...], block-causal, as in NCMGenerator). Each slot is
placed at the RoPE position of the text it summarises, so the pretrained model "sees"
the next chunk at the place where it would otherwise predict the next tokens.
The backbone is adapted with LoRA; slot inputs and per-level heads are trained fully.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .model import CoarseToFine, VQConfig


@dataclass
class QwenGenConfig:
    base: str = "Qwen/Qwen2.5-Coder-0.5B"
    max_prefix: int = 384          # Qwen tokens of context
    chunk_span: int = 128          # RoPE positions one chunk occupies
    lora_rank: int = 32


class QwenConceptGenerator(CoarseToFine):
    def __init__(self, cfg: QwenGenConfig, vq: VQConfig, lora=True):
        super().__init__()
        self.cfg, self.levels = cfg, list(vq.levels)
        self.tok = AutoTokenizer.from_pretrained(cfg.base)
        backbone = AutoModel.from_pretrained(cfg.base, dtype=torch.bfloat16, attn_implementation="sdpa")
        backbone.requires_grad_(False)
        if lora:
            from peft import LoraConfig, get_peft_model
            backbone = get_peft_model(backbone, LoraConfig(
                r=cfg.lora_rank, lora_alpha=cfg.lora_rank, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.backbone = backbone
        d = backbone.config.hidden_size
        std = backbone.get_input_embeddings().weight.float().std().item()
        n = sum(self.levels)
        self.inp = nn.Linear(vq.latent_dim, d)
        nn.init.normal_(self.inp.weight, std=std / vq.latent_dim ** 0.5)
        nn.init.zeros_(self.inp.bias)
        self.sos = nn.Parameter(torch.randn(1, self.levels[0], d) * std)
        self.slot = nn.Parameter(torch.randn(1, n, d) * std)          # which slot of which level
        self.heads = nn.ModuleList(nn.Linear(d, K) for K in vq.codebook_sizes)
        for h in self.heads:   # start from uniform predictions
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        group = torch.cat([torch.full((l,), k, dtype=torch.long) for k, l in enumerate(self.levels)])
        self.register_buffer("group", group, persistent=False)
        # slot j of level k sits at the middle of the text span it covers
        rel = torch.cat([((torch.arange(l) + 0.5) * cfg.chunk_span / l).long() for l in self.levels])
        self.register_buffer("rel_pos", rel, persistent=False)

    # ----- prefix handling
    def encode_prefix(self, texts, device):
        """Texts -> left-padded Qwen ids + mask, each prefixed with <|endoftext|> as BOS."""
        bos = self.tok.convert_tokens_to_ids("<|endoftext|>")
        rows = [[bos] + self.tok(t, add_special_tokens=False).input_ids[-(self.cfg.max_prefix - 1):] for t in texts]
        n = max(map(len, rows))
        ids = torch.tensor([[bos] * (n - len(r)) + r for r in rows], device=device)
        mask = torch.tensor([[0] * (n - len(r)) + [1] * len(r) for r in rows], device=device)
        return ids, mask

    def batch_info(self, prefix):
        return prefix[0].size(0), prefix[0].device

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if "lora_" in k or not k.startswith("backbone.")}

    # ----- model
    def forward(self, prefix, level_inputs):
        ids, pmask = prefix
        B, P = ids.shape
        emb = self.backbone.get_input_embeddings()(ids)
        parts = [self.sos.expand(B, -1, -1).to(emb.dtype)] + [self.inp(li.to(self.inp.weight.dtype)).to(emb.dtype)
                                                            for li in level_inputs]
        c = torch.cat(parts, dim=1)
        C = c.size(1)
        c = c + self.slot[:, :C].to(emb.dtype)
        x = torch.cat([emb, c], dim=1)

        n_real = pmask.sum(1, keepdim=True)
        ppos = (pmask.cumsum(1) - 1).clamp(min=0)
        pos = torch.cat([ppos, n_real + self.rel_pos[:C][None]], dim=1)

        # [prefix: causal, ignores padding] [slots: all prefix + slots of same or coarser level]
        g = self.group[:C]
        S = P + C
        allow = torch.zeros(B, S, S, dtype=torch.bool, device=ids.device)
        allow[:, :P, :P] = torch.ones(P, P, dtype=torch.bool, device=ids.device).tril()
        allow[:, :, :P] &= pmask[:, None, :].bool()
        allow[:, P:, P:] = g[:, None] >= g[None, :]
        allow |= torch.eye(S, dtype=torch.bool, device=ids.device)   # padded rows keep one key
        bias = torch.zeros(B, 1, S, S, dtype=emb.dtype, device=ids.device)
        bias.masked_fill_(~allow[:, None], torch.finfo(emb.dtype).min)

        h = self.backbone(inputs_embeds=x, attention_mask=bias, position_ids=pos).last_hidden_state[:, P:]
        logits, start = [], 0
        for k in range(1 + len(level_inputs)):
            l = self.levels[k]
            logits.append(self.heads[k](h[:, start:start + l].to(self.heads[k].weight.dtype)))
            start += l
        return logits
