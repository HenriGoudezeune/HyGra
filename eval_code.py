"""Measure how usable generated code is: syntax validity and functional correctness.

Each task is a signature + docstring (HumanEval-style). The model writes the body; the
completion is cut where the function ends, then checked with ast.parse and with asserts
run in a subprocess with a timeout.
"""
import argparse
import os
import ast
import subprocess
import sys

import torch

from ncm.data import get_encoder
from ncm.model import GenConfig, NCMGenerator, VQConfig, VQVAE

TASKS = [
    ('def add(a, b):\n    """Return the sum of a and b."""\n', "assert add(2, 3) == 5"),
    ('def is_even(n):\n    """Return True if n is even."""\n', "assert is_even(4) and not is_even(7)"),
    ('def factorial(n):\n    """Return n! for n >= 0."""\n', "assert factorial(5) == 120 and factorial(0) == 1"),
    ('def fibonacci(n):\n    """Return the n-th Fibonacci number (fibonacci(0) == 0)."""\n',
     "assert [fibonacci(i) for i in range(7)] == [0, 1, 1, 2, 3, 5, 8]"),
    ('def reverse_string(s):\n    """Return s reversed."""\n', "assert reverse_string('abc') == 'cba'"),
    ('def max_of_list(items):\n    """Return the largest element of a non-empty list."""\n',
     "assert max_of_list([3, 9, 2]) == 9"),
    ('def count_vowels(s):\n    """Return the number of vowels in s."""\n', "assert count_vowels('hello world') == 3"),
    ('def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards."""\n',
     "assert is_palindrome('racecar') and not is_palindrome('abc')"),
    ('def flatten(lists):\n    """Flatten a list of lists into one list."""\n',
     "assert flatten([[1, 2], [3], []]) == [1, 2, 3]"),
    ('def unique(items):\n    """Return the unique elements of items, keeping their order."""\n',
     "assert unique([1, 2, 1, 3, 2]) == [1, 2, 3]"),
    ('def read_lines(path):\n    """Return the lines of the text file at path, without newlines."""\n', None),
    ('def word_count(text):\n    """Return a dict mapping each word in text to how often it occurs."""\n',
     "assert word_count('a b a') == {'a': 2, 'b': 1}"),
    ('class Stack:\n    """A simple LIFO stack."""\n\n    def __init__(self):\n', None),
    ('def merge_dicts(a, b):\n    """Return a new dict with the keys of a and b; b wins on conflicts."""\n',
     "assert merge_dicts({'x': 1}, {'x': 2, 'y': 3}) == {'x': 2, 'y': 3}"),
    ('def clamp(x, lo, hi):\n    """Clamp x to the interval [lo, hi]."""\n',
     "assert clamp(5, 0, 3) == 3 and clamp(-1, 0, 3) == 0 and clamp(2, 0, 3) == 2"),
    ('def gcd(a, b):\n    """Return the greatest common divisor of a and b."""\n', "assert gcd(12, 18) == 6"),
]


def cut_function(prompt, completion):
    """Keep completion lines until a non-empty line dedents to the prompt's top level."""
    base = len(prompt) - len(prompt.lstrip())
    kept = []
    for line in completion.split("\n"):
        if line.strip() and (len(line) - len(line.lstrip())) <= base and kept:
            break
        kept.append(line)
    return prompt + "\n".join(kept).rstrip() + "\n"


def load(args, dev):
    ck = torch.load(args.tokenizer, map_location=dev)
    vcfg = VQConfig(**ck["cfg"])
    vq = VQVAE(vcfg).to(dev).eval()
    vq.load_state_dict(ck["model"])
    gk = torch.load(args.generator, map_location=dev)
    if gk.get("kind") == "qwen":
        from ncm.qwen_gen import QwenConceptGenerator, QwenGenConfig
        gen = QwenConceptGenerator(QwenGenConfig(**gk["cfg"]), vcfg).to(dev).eval()
        gen.load_state_dict(gk["model"], strict=False)
    else:
        gen = NCMGenerator(GenConfig(**gk["cfg"]), vcfg).to(dev).eval()
        gen.load_state_dict(gk["model"])
    return vq, gen


def load_decoder(path, vq, dev):
    """Context-reading decoder (train_decoder.py), or None to use the stage-1 decoder."""
    if not path or not os.path.exists(path):
        return None
    from ncm.context_decoder import ContextDecoder
    dec = ContextDecoder(vq).to(dev).eval()
    dec.load_state_dict(torch.load(path, map_location=dev)["model"])
    return dec


def next_chunk(vq, gen, enc, context, temperature, top_k, dev, decoder=None):
    """Stage-1 BPE ids of the context -> ids of the next chunk."""
    if hasattr(gen, "encode_prefix"):   # Qwen backbone reads the text itself
        prefix = gen.encode_prefix([enc.decode(context)], dev)
    else:
        L, W = vq.cfg.context_length, gen.cfg.prefix_chunks * vq.cfg.context_length
        window = context[-W:]
        window = [enc.eot_token] * (W - len(window)) + window
        prefix = vq.encode_chunks(torch.tensor([window], device=dev))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        f_hat, cums = gen.generate(prefix, vq.quant, temperature, top_k)
        if decoder is None:
            return vq.decode(f_hat).argmax(-1)[0].tolist(), cums
        L = vq.cfg.context_length
        ctx = context[-L:]
        ctx = torch.tensor([[enc.eot_token] * (L - len(ctx)) + ctx], device=dev)
        return decoder(f_hat, ctx).argmax(-1)[0].tolist(), cums


@torch.no_grad()
def complete(vq, gen, enc, prompt, chunks, temperature, top_k, dev, decoder=None):
    eot = enc.eot_token
    context, out = enc.encode_ordinary(prompt), []
    for _ in range(chunks):
        new, _ = next_chunk(vq, gen, enc, context + out, temperature, top_k, dev, decoder)
        if eot in new:
            return enc.decode(out + new[:new.index(eot)])
        out += new
    return enc.decode(out)


def run_test(code, test):
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code + "\n" + test], capture_output=True, timeout=5)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="ckpt/tokenizer.pt")
    p.add_argument("--generator", default="ckpt/generator.pt")
    p.add_argument("--data", default="data")
    p.add_argument("--decoder", default="ckpt/decoder_ctx.pt", help="context decoder (used if present)")
    p.add_argument("--samples", type=int, default=5, help="samples per task (pass@k uses all)")
    p.add_argument("--chunks", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--show", type=int, default=3, help="print this many completions")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    vq, gen = load(args, dev)
    decoder = load_decoder(args.decoder, vq, dev)
    print(f"decoder: {'context' if decoder else 'stage-1'}")
    enc = get_encoder(args.data)

    n = valid = 0
    tested = solved = 0
    shown = 0
    for prompt, test in TASKS:
        task_solved = False
        for _ in range(args.samples):
            code = cut_function(prompt, complete(vq, gen, enc, prompt, args.chunks,
                                                 args.temperature, args.top_k, dev, decoder))
            n += 1
            try:
                ast.parse(code)
                valid += 1
                ok_syntax = True
            except SyntaxError:
                ok_syntax = False
            if test and ok_syntax and not task_solved:
                task_solved = run_test(code, test)
            if shown < args.show:
                print(f"----- sample (syntax {'OK' if ok_syntax else 'ERROR'}) -----\n{code}")
                shown += 1
        if test:
            tested += 1
            solved += task_solved
    print(f"syntax valid: {valid}/{n} = {valid / n:.0%}")
    print(f"tests passed (pass@{args.samples}): {solved}/{tested} = {solved / tested:.0%}")


if __name__ == "__main__":
    main()
