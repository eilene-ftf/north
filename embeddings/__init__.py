"""Lexical analysis, parsing, and encoding functions for FORTH.

Example:
```python
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path

from embeddings import (
    lex,
    encode,
    HeteroAssoc,
    AutoAssoc,
    Codebook,
    EncodingEnvironment,
    savez
)

dim = 100

# Read the file, write to buffer
with open("./examples/sample.forth", "r") as f:
    buffer = f.read()

# Lexical analysis
words = lex(words)

# Encoding
codebook = Codebook([], dim=100)
codebook.initialize()
assoc_mem = HeteroAssoc(dim)
cleanup_mem = AutoAssoc(dim)
for value in codebook.values():
    cleanup_mem.write(value)
enc_env = EncodingEnvironment(codebook, assoc_mem, cleanup_mem)
embeddings = encode(words, enc_env)

# Serialize the embeddings
with TemporaryDirectory() as outdir:
    outpath = Path(outdir)
    savez(outdir, embeddings, enc_env)

    # Retrieve the embeddings
    embeddings_npz = np.load(outpath / "embeddings.npz")
    embeddings = embeddings_npz["embeddings"]

    # Retrieve the heteroassociative memory
    assoc_mem_npz = np.load(outpath / "assoc_mem.npz")
    assoc_mem_addrs = assoc_mem_npz["A"]
    assoc_mem_pats = assoc_mem_npz["P"]

    # Retrieve the autoassociative memory
    cleanup_mem_npz = np.load(outpath / "cleanup_mem.npz")
    assoc_mem_weights = cleanup_mem_npz["W"]

    # Retrieve the codebook
    codebook_npz = np.load(outpath / "codebook.npz")
    print(codebook_npz.files) # ==> the symbols associated with each random vector
```

For more information about working specifically with the serialized versions
of the embeddings, see the relevant function definitions: `.encodings.savez`,
`.encodings.HeteroAssoc.savez`, `.encodings.AutoAssoc.savez`,
`.encodings.Codebook.savez`, and ``.encodings.EncodingEnvironment.savez``.
"""

from .encoding import *
from .lex import *
