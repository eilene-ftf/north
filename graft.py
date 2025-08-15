"""Module for *grafting* FORTH word embeddings with the Nengo Semantic
pointer system.
"""

import pathlib
import sys
import typing

import nengo_spa as spa
import numpy as np

from embeddings import (
    AutoAssoc,
    Codebook,
    EncodingEnvironment,
    HeteroAssoc,
    encode,
    get_name,
    lex,
)


def load_embeddings(
    embeddings_path: str | pathlib.Path, vocab: spa.Vocabulary
) -> spa.SemanticPointer:
    """Load embeddings at `embeddings_path` into vocabulary, returning the semantic
    pointer which represents the embeddings.

    Args:
        embeddings_path (str | pathlib.Path): Path to the embeddings directory.
        vocab spa.Vocabulary: The current vocabulary.

    Returns:
        The semantic pointer representation of the program as a list.

    Example:
    ```python
    from graft import load_embeddings
    import nengo
    import nengo_spa as spa

    dim = 256
    voc = spa.Vocabulary(dim)
    voc_items = ["S_PUSH", "S_POP", "S_DUMP", "S_CODE_ERR_STACKEMPTY", "S_WORD"]
    voc.populate("; ".join(voc_items))

    # specify the path to the serialized embeddings
    embeddings_path = f"./data/fruit_program_dim{dim}"
    test_program = load_embeddings(embeddings_path, voc)

    print(type(test_program)) # => spa.SemanticPointer

    ... # do what you want with the program
    ```
    """
    print(f"Loading EMBEDDINGS from {embeddings_path}", file=sys.stderr)
    embeddings_path = (
        embeddings_path
        if isinstance(embeddings_path, pathlib.Path)
        else pathlib.Path(embeddings_path)
    )

    codebook_path = embeddings_path / "codebook.npz"
    print(f"Accessing CODEBOOK at {codebook_path}", file=sys.stderr)
    print("    Inserting serialized elements into VOCAB", file=sys.stderr)

    codebook_file = np.load(codebook_path)
    codebook = {}
    for file in codebook_file.files:
        codebook[file] = codebook_file[file]

    for name, value in codebook.items():
        print(f"        Adding {name} to VOCAB", file=sys.stderr)
        vocab.add(name, value)

    print("    FINISHED loading CODEBOOK into VOCAB", file=sys.stderr)
    codebook_file.close()
    print("", file=sys.stderr)

    embeddings_path = embeddings_path / "embeddings.npz"
    print(f"Accessing EMBEDDINGS at {embeddings_path}", file=sys.stderr)

    embeddings_file = np.load(embeddings_path)
    embeddings = embeddings_file["embeddings"]
    embeddings_file.close()

    cb = Codebook([], embeddings.shape[-1])
    cb.data = codebook
    embeddings_name = get_name(embeddings, cb)
    # vocab.add(embeddings_name, embeddings)
    return spa.SemanticPointer(embeddings, vocab=vocab, name=embeddings_name)


def embed(
    src: str,
    vocab: spa.Vocabulary,
    integer_encoding_scheme: typing.Literal["list", "binary"],
    width: int,
) -> spa.SemanticPointer:
    """Embed ``src`` into a ``spa.SemanticPointer``.

    Args:
        src str: The source text.
        vocab spa.Vocabulary: The vocabulary used by the simulation.

    Returns:
        The semantic pointer embedding of the source file.
    """
    words = lex(src)

    codebook = Codebook([], dim=vocab.dimensions)
    assoc_mem = HeteroAssoc(vocab.dimensions)
    cleanup_mem = AutoAssoc(vocab.dimensions)
    for value in codebook.values():
        cleanup_mem.write(value)

    enc_env = EncodingEnvironment(codebook, assoc_mem, cleanup_mem)
    if integer_encoding_scheme == "list":
        embeddings = encode(words, enc_env)
    else:
        embeddings = encode(words, enc_env, "binary", vocab, width)

    for name, value in codebook.items():
        if name not in vocab:
            vocab.add(name, value)

    embeddings_name = get_name(embeddings, enc_env.codebook)
    return spa.SemanticPointer(embeddings, vocab=vocab, name=embeddings_name)
