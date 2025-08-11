import pathlib

import numpy as np

from forth import (
    AutoAssoc,
    Codebook,
    EncodingEnvironment,
    HeteroAssoc,
    encode,
    lex,
    savez,
    decode,
)

if __name__ == "__main__":
    with open("./examples/sample.forth", "r") as f:
        buff = f.read()
        print(buff)

    words = lex(buff)
    print()
    print(words)
    print()

    dim = 200
    codebook = Codebook([], dim=dim)
    codebook.initialize()

    assoc_mem = HeteroAssoc(codebook.dim)
    cleanup_mem = AutoAssoc(codebook.dim)
    for value in codebook.values():
        cleanup_mem.write(value)

    enc_env = EncodingEnvironment(codebook, assoc_mem, cleanup_mem)
    words_arr = encode(words, enc_env)

    dpath = pathlib.Path(f"./data/sample_program_dim{dim}")
    savez(dpath, words_arr, enc_env)

    npzfile = np.load(f"./data/sample_program_dim{dim}/embeddings.npz")
    print(npzfile.files)

    print(decode(npzfile["embeddings"], enc_env))
