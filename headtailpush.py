import nengo
import nengo_spa as spa
import numpy as np
from numpy.linalg import norm
from collections import UserDict

theta = 0.2     # threshold parameter
d = 128         # dimensionality
voc = spa.Vocabulary(d)


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
    voc.populate("LEFT; RIGHT; PHI; NIL; APPLE; BANANA; CHERRY")
    
    lis = cons(voc['CHERRY'], cons(voc["APPLE"], voc["BANANA"]))
    listail1 = cons(voc["APPLE"], voc["BANANA"])
    listail2 = cons(voc["BANANA"],voc["NIL"])
    
    voc.add(lis.name, lis.v)
    
    R_state = spa.State(voc, label="R_state")
    H_state = spa.State(voc, label="H_state")
    T_state = spa.State(voc, label="T_state")
    
    
    def test_t_state(t, x):
        x = spa.SemanticPointer(x)
        return [vcos(x, voc[s]) for s in ('LS_APPLE_LS_BANANA_NIL', 'LS_BANANA_NIL', 'NIL')]
    T_node_out = nengo.Node(test_t_state, size_in=d,size_out=3)
    nengo.Connection(T_state.output, T_node_out)
    
    # output state for the head (stack push)
    output_state = spa.State(voc, label="output_state")
    
    def gate_transmission(t,x):
        return x if t < 1.0 else np.zeros(d)
        
    transmission_gate = nengo.Node(gate_transmission, size_in=d, size_out=d)    
        
    #lis >> R_state
    list_node = nengo.Node(output=lis.v)
    nengo.Connection(list_node, transmission_gate)
    nengo.Connection(transmission_gate, R_state.input)

    noisy_h = spa.State(voc)
    noisy_t = spa.State(voc)
    
    #head and tail 
    (~voc['LEFT'] * R_state) >> noisy_h
    (~voc['RIGHT'] * R_state) >> noisy_t
    
    def dereference_func(t,x):
        closest_sp = voc._keys[np.argmax(voc.dot(x))]
        
        if closest_sp.startswith("Pointer_to_"):
            original_name = closest_sp[11:]
            return voc[original_name].v
        x = cleanup(spa.SemanticPointer(x))
        return x.v
    
    tail_cleanup = nengo.Node(dereference_func, size_in=d, size_out=d)
    
    nengo.Connection(noisy_h.output, H_state.input)
    nengo.Connection(noisy_t.output, tail_cleanup)
    nengo.Connection(tail_cleanup, T_state.input)
    
    trigger = nengo.Node(lambda t: 1.0 if (t % 2) < 0.1 else 0.0)
    
    edge_detector = RisingEdgeDetector(tau=0.04,bias=0)
    nengo.Connection(trigger, edge_detector.input)
    
    def compute_condition(t, x):
        R_vec = x[0:d]
        edge_detected = x[d]
        dot_val = np.dot(R_vec, voc['NIL'].v)
        not_nil = 1.0 if dot_val < 0.9 else 0.0
        return [not_nil * (edge_detected > 0.5)]
    
    condition_node = nengo.Node(compute_condition, size_in=d+1, size_out=1)
    nengo.Connection(R_state.output, condition_node[0:d])
    nengo.Connection(edge_detector.output, condition_node[d])
   
    def vcos_node(t, x):
        mag = norm(x[:d]) * norm(x[d:])
        if mag == 0: return 0
        
        return (x[:d].dot(x[d:]))/mag
        
    #print(vcos(0, np.concatenate([listail1.v, listail2.v])))
        
    stable = nengo.Node(size_out=1, size_in=2*d, output=vcos_node)
    nengo.Connection(R_state.output, stable[:d])
    nengo.Connection(R_state.output, stable[d:],synapse=0.4)
    
    threshold = 0.95
    threshold_node = nengo.Node(
        size_in=1,
        size_out=1,
        output=lambda t, x: 1 if x < threshold else -1
    )
    nengo.Connection(stable, threshold_node)
    
    def latch_func(t, x):
        condition = x[0]
        stable = x[1]
        if not hasattr(latch_func, 'state'):
            latch_func.state = 0  # 0 = unlatched, 1 = latched
            
        if stable < -0.5:
            latch_func.state = 0
        
        if condition > 0.5:
            latch_func.state = 1
            
        return [latch_func.state]
    
    latch_node = nengo.Node(latch_func, size_in=2, size_out=1)
    nengo.Connection(condition_node, latch_node[0])
    nengo.Connection(threshold_node, latch_node[1])

    def t_state_gate(t, x):
        new_t = x[0:d]
        current_t = x[d:2*d]
        latch = x[2*d]
        
        if latch > 0.5:
            return current_t
        else:
            return new_t
    
    t_gate_node = nengo.Node(t_state_gate, size_in=2*d+1, size_out=d)
    nengo.Connection(tail_cleanup, t_gate_node[0:d])
    nengo.Connection(T_state.output, t_gate_node[d:2*d])
    nengo.Connection(latch_node, t_gate_node[2*d])
    nengo.Connection(t_gate_node, T_state.input, synapse=0.01)
    
    def gate_output(t, x):
        H_vec = x[0:d]
        cond = x[d]
        return H_vec if cond > 0.5 else np.zeros(d)
    
    gate_node_output = nengo.Node(gate_output, size_in=d+1, size_out=d)
    nengo.Connection(H_state.output, gate_node_output[0:d])
    nengo.Connection(condition_node, gate_node_output[d])
    nengo.Connection(gate_node_output, output_state.input)
    
    def update_R(t, x):
        T_vec = x[0:d]
        R_vec = x[d:2*d]
        cond = x[2*d]
        return cleanup(spa.SemanticPointer(T_vec if cond > 0.5 else R_vec)).v
    
    gate_node_R = nengo.Node(update_R, size_in=2*d+1, size_out=d)
    nengo.Connection(T_state.output, gate_node_R[0:d])
    nengo.Connection(R_state.output, gate_node_R[d:2*d], synapse=0.01)
    nengo.Connection(condition_node, gate_node_R[2*d])
    nengo.Connection(gate_node_R, R_state.input, synapse=0.05)