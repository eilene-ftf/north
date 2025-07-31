import nengo
import nengo_spa as spa
import numpy as np
from numpy.linalg import norm
from collections import UserDict

theta = 0.2     # threshold parameter
d = 128         # dimensionality
voc = spa.Vocabulary(d)

s = []

def make_stack_in(stack):
    stopwatch = 0
    state = 0
    out = np.zeros(d)
    
    def stack_in(t, x, vocab=voc):
        nonlocal stopwatch, state, out
        sig = x[d]
        inp = spa.SemanticPointer(x[:d])
        if state == 0 and sig > 1-theta:
            state = 1
            stopwatch = t
            if vcos(inp, vocab['PUSH']) > theta:
                stack.append(cleanup(inp - vocab['PUSH']))
            elif vcos(inp, vocab['POP']) > theta:
                out = vocab['POP'].v
        if state == 1 and sig < theta and t > stopwatch + 0.2:
            state = 0
            out = np.zeros(d)
        # print([v.name for v in stack])
        return np.concatenate((out, [state]))
    
    return stack_in
    
def make_stack_out(stack):
    stopwatch = 0
    state = np.zeros(d)
    sigout = 0
    
    def stack_out(t, x, vocab=voc):
        nonlocal stopwatch, state, sigout
        sig = x[d]
        inp = spa.SemanticPointer(x[:d])
        if sigout == 0 and sig > 1-theta:
            sigout = 1
            stopwatch = t
            if vcos(inp, vocab['POP']) > theta:
                print([v.name for v in stack])
                if stack:
                    state = stack.pop().v
                else:
                    state = vocab['S_CODE_ERROR'].v
        if sigout == 1 and sig < theta and t > stopwatch + 0.2:
            sigout = 0
        return np.concatenate((state, [sigout]))
    
    return stack_out
    

# An object to contain all the base symbols in both their string and
# holographic vector representations
# A string s can be looked up, given an HRR h, for lexicon L, with L.reverse[h]
# Similarly, L[s] = h
class Lexicon(UserDict):
    def __init__(self, mapping=None, *args, **kwargs):
        self.reverse = {}
        super().__init__(mapping, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.reverse[value] = key

    def update(self, other=(), *args, **kwds):
        super().update(other, **kwds)
        self.reverse.update(other=(((v, k) for k, v in other),))
        self.reverse.update(other=(((v, k) for k, v in kwds.items()),))

# A heteroassociative memory object employed because tuples turn out to degrade
# rather quickly. When CONS is called, we return a pointer to the tuple in the
# associative memory, which is automatically dereferenced by CAR, CDR
# Attention must still be paid to the action of the pointer, as it is an HRR
# and it is uncorrelated with its associate, and using it rather than
# dereferencing it can lead to errors
class SimpleAssoc():
    def __init__(self, lexicon=Lexicon(), theta=0.2):
        self.A = Lexicon()
        self.lexicon = lexicon
        self.reverse = lexicon.reverse
        self.theta = theta

    def memorize(self, probe, trace):
        if trace not in self.A.reverse:
            self.A[probe] = trace
        else:
            probe = self.A.reverse[trace]

        return (probe, trace)

    def recall(self, probe):
        if probe in self.A:
            return self.A[probe]

        keys = list(self.A.keys())
        activations = [probe==HRR(data=k) for k in keys]
        nearest = np.argmax(activations)

        if (activations[nearest]) > self.theta:
            return self.A[keys[nearest]]
        else:
            return probe

assoc_memory = SimpleAssoc()

def normalize(x):
    mag = x.length()
    if mag == 0: return x
    return x/mag

def vcos(u, v):
    mag = u.length() * v.length()
    dot = u.dot(v)
    if mag == 0: return dot
    return dot/mag

def is_list(u, vocab=voc):
    return vcos(u, vocab['PHI'] + vocab['NIL']) > theta

def cleanup(u, vocab=voc):
    unrolled = list(vocab.items())
    dots = np.array([v.dot(u) for k, v in unrolled])
    if (dots > theta).any():
        return vocab[unrolled[np.argmax(dots)][0]]
    return voc['Zero']

def cons(h, t, vocab=voc, simple_assoc=assoc_memory):

    if is_list(t):
        if h.name not in vocab: vocab.add(h.name, h)
        if t.name not in vocab: vocab.add(t.name, t)
        if vcos(t, vocab['NIL']) < 0.2:
            if f"Pointer_to_{t.name}" not in vocab:
                newv = np.random.normal(size = d, scale=1/np.sqrt(d))
                newv_name = f"Pointer_to_{t.name}"
                vocab.add(newv_name, newv)
                assoc_memory.memorize(vocab[newv_name], t)
                #print(newv)
            t_name = f"Pointer_to_{t.name}"
            t = vocab[t_name]
            
            
            #print(list(assoc_memory.A.reverse.items()))
        new_sp = vocab['LEFT'] * h + vocab['RIGHT'] * t + vocab['PHI']
        new_name = f'LS_{h.name}_{t.name.replace("Pointer_to_", "")}'
        return spa.SemanticPointer(new_sp.normalized().v, name=new_name)
    return cons(h, cons(t, vocab['NIL']), vocab=vocab)

def car(l, vocab=voc):
    
    return cleanup(~vocab['LEFT'] * l, vocab=vocab)

def cdr(l, vocab=voc):
    p = cleanup(~vocab['RIGHT'] * l, vocab=vocab)
    if p.name == 'NIL':
        return p
    sliced_name = p.name[11:]
    print(sliced_name)
    return vocab[sliced_name]

def read(l, vocab=voc):

    lis = []
    while vcos(l, vocab['NIL']) < theta:
        lis.append(car(l).name)
        l = cdr(l)
    return lis
        
def make_list(lis, vocab=voc):

    l = vocab[lis.pop()]
    print(l)
    while lis:
        l = cons(vocab[lis.pop()], l)
    return l
    
class RisingEdgeDetector(nengo.Network):
    def __init__(self, tau=0.01, n_neurons=100, bias=-0.5, **kwargs):
        super().__init__(**kwargs)
        
        with self:
            self.input = nengo.Node(size_in=1)
            
            self.delayed = nengo.Node(size_in=1)
            nengo.Connection(self.input, self.delayed, synapse=tau)
            
            self.compare = nengo.Ensemble(n_neurons, 1)
            
            nengo.Connection(self.input, self.compare, transform=2)
            nengo.Connection(self.delayed, self.compare, transform=-2)
            nengo.Connection(nengo.Node(output=bias), self.compare)
            
            self.output = self.compare   


model = spa.Network()
with model:
    voc.populate("LEFT; RIGHT; PHI; NIL; APPLE; BANANA; CHERRY; PUSH; POP; S_CODE_ERROR")
    
    lis = cons(voc['CHERRY'], cons(voc["APPLE"], voc["BANANA"]))
    listail1 = cons(voc["APPLE"], voc["BANANA"])
    listail2 = cons(voc["BANANA"],voc["NIL"])
    
    voc.add(lis.name, lis.v)
    
    sigin = nengo.Node(0)
    sigout = nengo.Node(size_in=1)
    inp = spa.State(voc)
    out = spa.State(voc)
    
    stack_in = nengo.Node(size_in=d+1, output=make_stack_in(s))
    stack_out = nengo.Node(size_in=d+1, output=make_stack_out(s))
    
    nengo.Connection(inp.output, stack_in[:d])
    nengo.Connection(sigin, stack_in[d])
    nengo.Connection(stack_in, stack_out)
    nengo.Connection(stack_out[:d], out.input)
    nengo.Connection(stack_out[d], sigout)
