import nengo
import nengo_spa as spa
import numpy as np

from buffer import RingBuffer


model = spa.Network()
with model:
   d = 256
   s = 10
   vocab = spa.Vocabulary(d)
   vocab.populate("APPLE; BANANA; CHERRY; DURIAN; ELDERBERRY")
   pub = spa.Network()
   sub = spa.Network()
   buf = RingBuffer(buf_size=s, dim=d, pub=pub, sub=sub, sub_vocab=vocab, label="MyBuffer")
   #buffer2 = RingBuffer(buf_size=s, dim=d, pub=pub, sub=sub, label="MySecondBuffer")

