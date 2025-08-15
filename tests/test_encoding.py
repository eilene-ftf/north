import nengo_spa as spa  # type: ignore
import pytest

import embeddings as embed
import numerical as num
import graft


def test_bitstring_encoding() -> None:
    # TODO: implement bitstring encoding tests
    src = "1"
    
    dim = 256
    codebook = embed.Codebook([], dim=dim)
    codebook.initialize()
    cleanup_mem = embed.AutoAssoc(dim=dim)
    assoc_mem = embed.HeteroAssoc(dim_A=dim)
    for value in codebook.values():
        cleanup_mem.write(value)

    enc_env = embed.EncodingEnvironment(codebook, assoc_mem, cleanup_mem)
    vocab = spa.Vocabulary(dimensions=dim)

    words = embed.lex(src)

    embeddings = embed.encode(words, enc_env, integer_encoding_scheme="binary", vocab=vocab)

    graft.embed()