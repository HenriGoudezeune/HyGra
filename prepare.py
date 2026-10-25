"""Tokenize a code corpus into data/train.bin and data/val.bin.

Default corpus: the Python standard library of the interpreter running this script.
"""
import argparse
import sysconfig

from ncm.data import build_corpus

p = argparse.ArgumentParser()
p.add_argument("--src", nargs="*", default=[sysconfig.get_paths()["stdlib"]],
               help="directories to scan for source files")
p.add_argument("--ext", nargs="*", default=[".py"])
p.add_argument("--out", default="data")
args = p.parse_args()

build_corpus(args.src, args.out, exts=tuple(args.ext))
