# ncm-code

A small **Next-Concept Modeling (NCM)** model for code generation, trained locally on an RTX 3050 (4 GB).

Based on *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*
(Kumbong, Khare, Ré et al., Stanford / Hazy Research, arXiv 2610.14162,
[github.com/HazyResearch/ncm](https://github.com/HazyResearch/ncm)).
The official repo only ships evaluation code. This repo is a compact re-implementation with training code.

## Idea

NCM does not generate one token at a time. It does this instead:

1. **Tokenizer (stage 1).** A VQ-VAE turns a chunk of 128 GPT-2 BPE tokens into a 32-position latent. A *multiscale residual VQ* quantizes that latent into codes at the scales `1, 2, 4, 8, 16, 32`, from coarse ("what kind of code is this") to fine (exact tokens).
2. **Generator (stage 2).** A block-causal transformer predicts the codes of the next chunk one scale at a time. Within a scale, all codes are predicted in parallel. One chunk of 128 tokens therefore takes **6 forward passes**, where an autoregressive model would need 128.

## Usage

```powershell
uv sync                                   # torch with CUDA 12.8
uv run python prepare.py                  # corpus = Python stdlib (or --src <dir> ...)
uv run python train_tokenizer.py          # stage 1  -> ckpt/tokenizer.pt
uv run python train_generator.py          # stage 2  -> ckpt/generator.pt
uv run python generate.py --prompt "def quicksort(arr):`n" --show-levels
```

Watch `[val] token reconstruction acc` in stage 1. If it stays well below roughly 0.95, the generator can never produce clean code.

## Differences from the paper

- The decoder receives the quantized latent through a straight-through estimator, so the encoder trains on reconstruction loss. In the original default, the encoder only receives the commitment loss.
- The model is tiny (about 16M + 15M parameters) and trains on little data. Expect output that looks like Python, not working programs.
- There is no classifier-free guidance and no KV cache. With sequences of at most 95 positions, neither is needed.
