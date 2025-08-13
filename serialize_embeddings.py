"""Serialize the embeddings of a FORTH source file.

Run `python serialize_embeddings.py -h` for more information.
"""

import argparse as ap
import pathlib
import sys

import numpy as np

from embeddings import (
    AutoAssoc,
    Codebook,
    EncodingEnvironment,
    HeteroAssoc,
    encode,
    lex,
    savez,
)


def main() -> None:
    parser = ap.ArgumentParser(
        prog="serialize_embeddings",
        description="Script for serialization of HRR embeddings for FORTH programs",
        epilog="🐈🐱",
    )
    parser.add_argument("--src", type=str, help="Path to the FORTH source file.")
    parser.add_argument(
        "--out_dir", "-o", type=str, help="Path to directory to store embeddings"
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        help="Dimensionality of the high-dimensional vectors used in the embedding.",
    )
    args = parser.parse_args()

    if args.src is None:
        print("PLEASE SPECIFY ``SRC`` FILE!", file=sys.stderr)
        parser.print_help()
        return

    if args.out_dir is None:
        print("PLEASE SPECIFY ``OUT_DIR`` DIRECTORY!", file=sys.stderr)
        parser.print_help()
        return

    if args.dim is None:
        print("PLEASE SPECIFY ``OUT_DIR`` DIRECTORY!", file=sys.stderr)
        parser.print_help()
        return

    with open(args.src, "r") as f:
        buff = f.read()
        print(buff)

    words = lex(buff)
    print()
    print([word.cont for word in words])
    print()

    dim = args.dim
    codebook = Codebook([], dim=dim)
    codebook.initialize()

    assoc_mem = HeteroAssoc(codebook.dim)
    cleanup_mem = AutoAssoc(codebook.dim)
    for value in codebook.values():
        cleanup_mem.write(value)

    enc_env = EncodingEnvironment(codebook, assoc_mem, cleanup_mem)
    words_arr = encode(words, enc_env)

    dpath = pathlib.Path(args.out_dir)
    savez(dpath, words_arr, enc_env)


if __name__ == "__main__":
    main()
