# HyGra · Next-Concept Modeling on a laptop

<img src="web/logo.png" width="96" align="right" alt="HyGra logo">

**HyGra** (from *HydrarGyrum*, Hg, element 80) is a desktop chat app plus an experiment.
It tries to train a **Next-Concept Model (NCM)** for Python code on a single 4 GB laptop GPU (RTX 3050).
The app also talks to **Claude Opus 5.5** through Claude Code and your Claude subscription, so it is useful while the experiment is not.

Based on *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*
(Kumbong, Khare, Ré et al., Stanford / Hazy Research, arXiv 2610.14162,
[github.com/HazyResearch/ncm](https://github.com/HazyResearch/ncm)).
The official repo ships evaluation code only. This repo is a compact re-implementation with training code.

> **Honest summary:** the concept models here are **fast but not useful**. They produce one 128-token chunk in 73 ms, where plain token-by-token generation takes seconds. But only 1–2% of their output is valid Python. See [Results](#results).

## How NCM works

1. **Tokenizer (stage 1).** A VQ-VAE turns a chunk of 128 BPE tokens into a 64-position latent. A *multiscale residual VQ* quantizes the latent at the scales `1, 2, 4, 8, 16, 32, 48, 64, 64`, from coarse ("what kind of code is this") to fine (exact tokens).
2. **Generator (stage 2).** The generator predicts the next chunk's codes one scale at a time, and all codes of a scale in parallel. A 128-token chunk therefore takes **9 forward passes**, where an autoregressive model needs 128.

This repo has three variants:

| | What | Code |
|---|---|---|
| **NCM 34M** | Generator trained from scratch (block-causal transformer) | `train_generator.py` |
| **Qwen-NCM 0.5B** | Qwen2.5-Coder-0.5B as backbone (LoRA). It reads the text, and concept slots sit at the RoPE positions of the text they summarize | `ncm/qwen_gen.py`, `train_generator_qwen.py` |
| **Context decoder** | A VQ decoder that also reads the previous chunk. The codes are frozen, so it works with both generators | `ncm/context_decoder.py`, `train_decoder.py` |

## Results

Evaluation setup:
- **Hardware:** RTX 3050 Laptop, 4 GB.
- **Data:** 409M tokens of filtered Python from `codeparrot/codeparrot-clean`, with an own 16k BPE vocabulary.
- **Tests:** 16 small function tasks in `eval_code.py`, 5 samples each.

| | NCM 34M | Qwen-NCM 0.5B |
|---|---|---|
| Training | 20k steps × 32 (3.3 h) | 8k steps × 8 (6.5 h) |
| Valid Python (stage-1 decoder) | 1% | 0% |
| Valid Python (context decoder) | 2% | 0% |
| Tests passed (pass@5) | 0 / 14 | 0 / 14 |
| Coarsest level: loss (uniform = 5.55) | 4.61 | 5.45 |
| Finest level: accuracy | 33% | 6% |
| **Time per 128-token chunk** | **73 ms** | 726 ms |

For comparison, Qwen2.5-Coder-0.5B generating token by token took 5203 ms per 128 tokens on the same GPU (`bench_speed.py`).

Stage-1 token reconstruction is 95.7%. The context decoder raises it to 96.5%.

What we learned:
- **Speed:** the speed-up of concept generation is real.
- **Coarse levels:** these are the problem. The generators barely beat the unigram entropy on levels 0–6. Every 128-token chunk is then decoded from a poor "plan", and the output falls apart.
- **Pretrained backbone:** it does *not* fix the coarse levels at this budget. In the same wall-clock time it sees 4× fewer examples, and it ends up worse.
- **Reconstruction errors:** 1 wrong token in about 28 already breaks syntax. Even a perfect generator would need a better tokenizer.

## The app

<img src="web/logo.png" width="20"> **HyGra** is a native Windows window (pywebview / WebView2):
- **Design:** olive green, thick angular orange lines.
- **Models:** Claude Opus 5.5 for real conversations, or the local NCM models.
- **Watching the concept models:** you can see them write level by level, and inspect what each level "thinks".

```powershell
uv sync
uv run python install_shortcut.py   # icon + Start-menu / desktop shortcut "HyGra"
uv run python app.py                # or start it from the Start menu
```

- **Claude:** runs through the [Claude Code](https://claude.com/claude-code) CLI, logged in with your Claude Pro/Max subscription (run `claude` once in a terminal to log in). No API key and no extra cost: chats count toward your plan's normal usage limits. Each HyGra chat is one Claude Code session with all tools switched off, so Claude remembers the conversation but cannot touch your files. An `ANTHROPIC_API_KEY` in the environment is deliberately ignored, so it never falls back to paid API billing.
- **Switching models:** Claude and the local GPU models run side by side; you can switch model or chat while an answer is still coming in.
- **Local models:** need the checkpoints in `ckpt/` (see below).

## Training it yourself

```powershell
uv sync                                    # torch with CUDA 12.8
uv run python prepare.py                   # download + filter codeparrot, train BPE -> data/
uv run python train_tokenizer.py           # stage 1      -> ckpt/tokenizer.pt
uv run python train_generator.py           # stage 2      -> ckpt/generator.pt
uv run python train_generator_qwen.py      # stage 2 alt. -> ckpt/generator_qwen.pt
uv run python train_decoder.py             # stage 1b     -> ckpt/decoder_ctx.pt
uv run python eval_code.py --generator ckpt/generator.pt
uv run python diagnose_levels.py --generator ckpt/generator.pt
uv run python bench_speed.py
uv run python generate.py --prompt "def quicksort(arr):`n" --show-levels
```

**GPU temperature:**
- **Pausing:** every training script pauses when the GPU goes above 82 °C, and resumes below 72 °C (`ncm/thermal.py`).
- **Watchdog:** `gpu_watchdog.ps1` logs the temperature and stops training at ≥ 87 °C.

## Checkpoints

The trained weights are attached to the GitHub release:
- `tokenizer.pt`
- `generator.pt`
- `generator_qwen.pt`, which contains only the LoRA adapters and the new layers. The base model is downloaded from Hugging Face.
- `decoder_ctx.pt`

Put them in `ckpt/`. `data/bpe.json` is in the release too.

## Licenses

- **This code:** MIT, see `LICENSE`.
- **Qwen-NCM:** built on [Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) (Apache 2.0).
- **Training data:** `codeparrot/codeparrot-clean`. Its source files carry their own licenses, so the data itself is not redistributed here.
