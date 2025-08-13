import nengo
import nengo_spa as spa
import numpy as np

from buffer import RingBuffer


model = spa.Network()
with model:
   d = 256
   s = 10
   pub = spa.Network()
   sub = spa.Network()
   buffer = RingBuffer(buf_size=s, dim=d, pub=pub, sub=sub, label="MyBuffer")
   #buffer2 = RingBuffer(buf_size=s, dim=d, pub=pub, sub=sub, label="MySecondBuffer")

