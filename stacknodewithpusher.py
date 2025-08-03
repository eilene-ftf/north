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
                if stack:
                    state = stack.pop().v
                else:
                    state = vocab['S_CODE_ERROR'].v
        if sigout == 1 and t > stopwatch + 0.2:
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

def dereference(t, x):
    closest_sp = voc._keys[np.argmax(voc.dot(x))]
    if closest_sp.startswith("Pointer_to_"):
        return voc[closest_sp[11:]].v
    return cleanup(spa.SemanticPointer(x), vocab=voc).v            
            
class Pusher(spa.Network):
    def __init__(self, d=128, theta=0.2, items=None, label="Pusher"):
        super().__init__(label="Pusher")
        self.d = d
        self.theta = theta
        self.voc = spa.Vocabulary(d)
        self.voc.populate("LEFT; RIGHT; PHI; NIL")
        self.assoc_memory = SimpleAssoc(theta=theta)
        
        self.vector_numbers = items.v
            
        with self:
            # Labeled states
            R_state = spa.State(self.voc, label="R_register")
            H_state = spa.State(self.voc, label="head_register")
            T_state = spa.State(self.voc, label="tail_register")
            output_state = spa.State(self.voc, label="output_register")
            
            self.input_node = nengo.Node(output=self.vector_numbers, label="input_register")
            self.condition_output = nengo.Node(size_in=1, label="condition_flag")
            self.head_output = output_state
            
            def gate_transmission(t,x):
                return x if t < 1.0 else np.zeros(self.d)
                
            # Labeled nodes
            transmission_gate = nengo.Node(
                gate_transmission, 
                size_in=self.d, 
                size_out=self.d,
                label="input_gate"
            )
                
            nengo.Connection(self.input_node, transmission_gate)
            nengo.Connection(transmission_gate, R_state.input)

            noisy_h = spa.State(self.voc, label="noisy_head")
            noisy_t = spa.State(self.voc, label="noisy_tail")
            
            (~self.voc['LEFT'] * R_state) >> noisy_h
            (~self.voc['RIGHT'] * R_state) >> noisy_t
            
            tail_cleanup = nengo.Node(
                dereference, 
                size_in=self.d, 
                size_out=self.d,
                label="tail_dereference"
            )
            
            nengo.Connection(noisy_h.output, H_state.input)
            nengo.Connection(noisy_t.output, tail_cleanup)
            nengo.Connection(tail_cleanup, T_state.input)
            
            self.trigger = nengo.Node(
                lambda t: 1.0 if (t % 2) < 0.1 else 0.0, 
                label="clock_trigger"
            )
            
            edge_detector = RisingEdgeDetector(tau=0.04, bias=0, label="edge_detector")
            nengo.Connection(self.trigger, edge_detector.input)
            
            def compute_condition(t, x):
                R_vec = x[0:self.d]
                edge_detected = x[self.d]
                dot_val = np.dot(R_vec, self.voc['NIL'].v)
                not_nil = 1.0 if dot_val < 0.9 else 0.0
                return [not_nil * (edge_detected > 0.5)]
            
            condition_node = nengo.Node(
                compute_condition, 
                size_in=self.d+1, 
                size_out=1,
                label="nil_detector"
            )
            nengo.Connection(R_state.output, condition_node[0:self.d])
            nengo.Connection(edge_detector.output, condition_node[self.d])
            nengo.Connection(condition_node, self.condition_output)
       
            def vcos_node(t, x):
                mag = norm(x[:self.d]) * norm(x[self.d:])
                if mag == 0: return 0
                return (x[:self.d].dot(x[self.d:]))/mag
                
            stable = nengo.Node(
                size_out=1, 
                size_in=2*self.d, 
                output=vcos_node,
                label="stability_detector"
            )
            nengo.Connection(R_state.output, stable[:self.d])
            nengo.Connection(R_state.output, stable[self.d:], synapse=0.4)
            
            threshold = 0.95
            threshold_node = nengo.Node(
                size_in=1,
                size_out=1,
                output=lambda t, x: 1 if x < threshold else -1,
                label="stability_threshold"
            )
            nengo.Connection(stable, threshold_node)
            
            def latch_func(t, x):
                condition = x[0]
                stable = x[1]
                if not hasattr(latch_func, 'state'):
                    latch_func.state = 0
                if stable < -0.5:
                    latch_func.state = 0
                if condition > 0.5:
                    latch_func.state = 1
                return [latch_func.state]
            
            latch_node = nengo.Node(
                latch_func, 
                size_in=2, 
                size_out=1,
                label="update_latch"
            )
            nengo.Connection(condition_node, latch_node[0])
            nengo.Connection(threshold_node, latch_node[1])

            def t_state_gate(t, x):
                new_t = x[0:self.d]
                current_t = x[self.d:2*self.d]
                latch = x[2*self.d]
                if latch > 0.5:
                    return current_t
                else:
                    return new_t
            
            t_gate_node = nengo.Node(
                t_state_gate, 
                size_in=2*self.d+1, 
                size_out=self.d,
                label="tail_register_gate"
            )
            nengo.Connection(tail_cleanup, t_gate_node[0:self.d])
            nengo.Connection(T_state.output, t_gate_node[self.d:2*self.d])
            nengo.Connection(latch_node, t_gate_node[2*self.d])
            nengo.Connection(t_gate_node, T_state.input, synapse=0.01)
            
            nengo.Connection(H_state.output, output_state.input)
            
            def update_R(t, x):
                T_vec = x[0:self.d]
                R_vec = x[self.d:2*self.d]
                cond = x[2*self.d]
                return cleanup(spa.SemanticPointer(T_vec if cond > 0.5 else R_vec), vocab=self.voc).v
            
            gate_node_R = nengo.Node(
                update_R, 
                size_in=2*self.d+1, 
                size_out=self.d,
                label="R_register_gate"
            )
            nengo.Connection(T_state.output, gate_node_R[0:self.d])
            nengo.Connection(R_state.output, gate_node_R[self.d:2*self.d], synapse=0.01)
            nengo.Connection(condition_node, gate_node_R[2*self.d])
            nengo.Connection(gate_node_R, R_state.input, synapse=0.05)


def create_modification_node(vocab, theta=0.2):
    d = vocab.dimensions
    pop_vec = vocab["POP"].v
    push_vec = vocab["PUSH"].v
    
    def modify_output(t, x):
        output_vec = x[:d]
        
        norm_output = np.linalg.norm(output_vec)
        norm_pop = np.linalg.norm(pop_vec)
        cos_sim = 0
        if norm_output > 1e-6 and norm_pop > 1e-6:
            cos_sim = np.dot(output_vec, pop_vec) / (norm_output * norm_pop)
        
        if cos_sim > theta:
            return pop_vec
        else:
            combined = push_vec + output_vec
            norm_combined = np.linalg.norm(combined)
            if norm_combined > 1e-6:
                combined /= norm_combined
            return combined
    
    return modify_output

model = spa.Network()
with model:
    voc.populate("LEFT; RIGHT; PHI; NIL; APPLE; BANANA; CHERRY; PUSH; POP; S_CODE_ERROR")
    
    lis = cons(voc['CHERRY'], cons(voc["APPLE"], voc["BANANA"])) #Putting  this
    listail1 = cons(voc["APPLE"], voc["BANANA"])
    listail2 = cons(voc["BANANA"],voc["NIL"])
    
    voc.add(lis.name, lis.v)
    
    sigin = nengo.Node(size_in=1)
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
    
    pusher = Pusher(d=d, theta=theta, items=lis, label="Pusher Network") #Into this
    
    mod_node = nengo.Node(
        create_modification_node(voc, theta=theta),
        size_in=voc.dimensions,
        size_out=voc.dimensions
    )
    
    nengo.Connection(pusher.head_output.output, mod_node)
    
    nengo.Connection(mod_node, inp.input)
    
    nengo.Connection(pusher.trigger, sigin)