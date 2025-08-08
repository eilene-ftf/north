import nengo
import nengo_spa as spa
from nengo_spa.connectors import RoutedConnection

import numpy as np
from numpy.linalg import norm
from collections import UserDict

theta = 0.3     # threshold parameter
d = 256         # dimensionality
voc = spa.Vocabulary(d)


def make_stack_in(stack):
    stopwatch = 0
    state = 0
    out = np.zeros(d)
    
    def stack_in(t, x, vocab=voc):
        nonlocal stopwatch, state, out
        if t < 0.1 and stack:
            del stack[:]
            print([p.name for p in stack])
        sig = x[d]
        inp = spa.SemanticPointer(x[:d])
        if state == 0 and sig > 1-theta:
            state = 1
            stopwatch = t
            if vcos(inp, vocab['S_PUSH']) > theta:
                stack.append(cleanup(inp - vocab['S_PUSH']))
                print([p.name for p in stack])
            elif vcos(inp, vocab['S_POP']) > theta:
                out = vocab['S_POP'].v
                print([p.name for p in stack])
            elif vcos(inp, vocab['S_DUMP']) > theta:
                del stack[:]
                print([p.name for p in stack])
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
            if vcos(inp, vocab['S_POP']) > theta:
                if stack:
                    state = stack.pop().v
                else:
                    state = vocab['S_CODE_ERR_STACKEMPTY'].v
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

assoc = SimpleAssoc()

def make_list(lis, vocab=voc, assoc_memory=assoc):
    vector_list = vocab['T_NIL']
    
    for item in lis[::-1]:
        vector_list = cons(vocab[item], vector_list, vocab=voc, assoc_memory=assoc_memory)
        
    return vector_list

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
    return vcos(u, vocab['R_PHI'] + vocab['T_NIL']) > theta

def cleanup(u, vocab=voc):
    unrolled = list(vocab.items())
    dots = np.array([v.dot(u) for k, v in unrolled])
    if (dots > theta).any():
        return vocab[unrolled[np.argmax(dots)][0]]
    return voc['Zero']

def cons(h, t, vocab=voc, assoc_memory=assoc):

    if is_list(t):
        if h.name not in vocab: vocab.add(h.name, h)
        if t.name not in vocab: vocab.add(t.name, t)
        if vcos(t, vocab['T_NIL']) < 0.2:
            if f"Pointer_to_{t.name}" not in vocab:
                newv = np.random.normal(size = d, scale=1/np.sqrt(d))
                newv_name = f"Pointer_to_{t.name}"
                vocab.add(newv_name, newv)
                assoc_memory.memorize(vocab[newv_name], t)
                #print(newv)
            t_name = f"Pointer_to_{t.name}"
            t = vocab[t_name]
            
            
            #print(list(assoc_memory.A.reverse.items()))
        new_sp = vocab['R_LEFT'] * h + vocab['R_RIGHT'] * t + vocab['R_PHI']
        new_name = f'LS_{h.name}_{t.name.replace("Pointer_to_", "")}'
        return spa.SemanticPointer(new_sp.normalized().v, name=new_name)
    return cons(h, cons(t, vocab['T_NIL']), vocab=vocab)

def car(l, vocab=voc):
    
    return cleanup(~vocab['R_LEFT'] * l, vocab=vocab)

def cdr(l, vocab=voc):
    p = cleanup(~vocab['R_RIGHT'] * l, vocab=vocab)
    if p.name == 'T_NIL':
        return p
    sliced_name = p.name[11:]
    #print(sliced_name)
    return vocab[sliced_name]

def read(l, vocab=voc):

    lis = []
    while vcos(l, vocab['T_NIL']) < theta:
        lis.append(car(l).name)
        l = cdr(l)
    return lis
        
def make_list(lis, vocab=voc):

    l = vocab[lis.pop()]
    #print(l)
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
            nengo.Connection(nengo.Node(output=bias, label="bias"), self.compare)
            
            self.output = nengo.Node(size_in=1)
            nengo.Connection(self.compare, self.output)

def dereference(t, x):
    closest_sp = voc._keys[np.argmax(voc.dot(x))]
    if closest_sp.startswith("Pointer_to_"):
        return voc[closest_sp[11:]].v
    return cleanup(spa.SemanticPointer(x), vocab=voc).v            
            
class ControlUnit(spa.Network):
    def __init__(self, circuits, d=d, theta=theta, items=None, label="ControlUnit", voc=spa.Vocabulary(d), assoc_memory=assoc):
        super().__init__(label=label)
        self.d = d
        self.theta = theta
        self.voc = voc
        to_add = [l for l in ['R_LEFT', 'R_RIGHT', 'R_PHI', 'T_NIL'] if l not in voc]
        voc.populate(';'.join(to_add))
        self.assoc_memory = assoc_memory
        
        self.vector_numbers = items.v
            
        with self:
            # Labeled states
            R_state = spa.State(self.voc, label="R_register")
            H_state = spa.State(self.voc, label="head_register")
            T_state = spa.State(self.voc, label="tail_register")
            output_state = spa.State(self.voc, label="output_register")
            
            mod_node = nengo.Node(
                create_modification_node(voc, circuits=circuits, theta=theta),
                size_in=voc.dimensions+1,
                size_out=2*voc.dimensions+1,
                label="mod_node"
            )
            
            self.output = mod_node
            
            self.is_paused = nengo.Node(size_in=1, label="paused?")
            nengo.Connection(mod_node[-1], self.is_paused)

            self.to_data_stack = nengo.Node(size_in=d, label="to_data_stack")
            nengo.Connection(mod_node[:d], self.to_data_stack)

            self.to_dispatcher = nengo.Node(size_in=d, label="to_dispatcher")
            nengo.Connection(mod_node[d:d*2], self.to_dispatcher)
            
            nengo.Connection(output_state.output, mod_node[:d])
            
            self.input_node = nengo.Node(output=self.vector_numbers, label="input_node")
            self.word_busy = nengo.Node(size_in=1, label="word_busy")
            self.condition_output = nengo.Node(size_in=1, label="condition_flag")
            
            # May need to be changed, seems like it might behave weirdly if 
            # there are multiple words in a row or processing a word takes a 
            # long time
            resume = RisingEdgeDetector(tau=1.5, bias=0, label="resume")
            nengo.Connection(self.word_busy, resume.input)
            nengo.Connection(resume.output, mod_node[-1])
            
            def cleanup_node(t, x, vocab=self.voc):
                return cleanup(x, vocab=voc).v
            
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
            
            (~self.voc['R_LEFT'] * R_state) >> noisy_h
            (~self.voc['R_RIGHT'] * R_state) >> noisy_t
            
            tail_cleanup = nengo.Node(
                dereference, 
                size_in=self.d, 
                size_out=self.d,
                label="tail_dereference"
            )
            
            head_cleanup = nengo.Node(size_in=d, size_out=d, output=cleanup_node, label="head_cleanup")
            
            nengo.Connection(noisy_h.output, head_cleanup)
            nengo.Connection(head_cleanup, H_state.input)
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
                pause = x[self.d + 1]
                dot_val = np.dot(R_vec, self.voc['T_NIL'].v)
                not_nil = 1.0 if dot_val < 0.9 else 0.0
                return [not_nil * (edge_detected > 0.5) * (pause < theta)]
            
            condition_node = nengo.Node(
                compute_condition, 
                size_in=self.d+2, 
                size_out=1,
                label="nil_detector"
            )
            nengo.Connection(R_state.output, condition_node[0:self.d])
            nengo.Connection(edge_detector.output, condition_node[self.d])
            nengo.Connection(condition_node, self.condition_output)
            nengo.Connection(mod_node[-1], condition_node[-1])
            
       
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
                pause = x[-1]
                if latch > 0.5 or pause > theta:
                    return current_t
                else:
                    return new_t
            
            t_gate_node = nengo.Node(
                t_state_gate, 
                size_in=2*self.d+2, 
                size_out=self.d,
                label="tail_register_gate"
            )
            nengo.Connection(tail_cleanup, t_gate_node[0:self.d])
            nengo.Connection(T_state.output, t_gate_node[self.d:2*self.d])
            nengo.Connection(latch_node, t_gate_node[2*self.d])
            nengo.Connection(mod_node[-1], t_gate_node[-1])
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
            
            
            

class SimpleStack(spa.Network):
    def __init__(self, d=d, label="stack memory"):
        super().__init__(label=label)
        self.d = d
        self.stack = []
        
        with self:
            stack_in = nengo.Node(size_in=self.d+1, output=make_stack_in(self.stack), label="stack_in")
            stack_out = nengo.Node(size_in=self.d+1, output=make_stack_out(self.stack), label="stack_out")
            
            self.sigin = nengo.Node(size_in=1, label="sigin")
            self.sigout = nengo.Node(size_in=1, label="sigout")
            
            self.input = nengo.Node(size_in=self.d, label="input")
            self.output = nengo.Node(size_in=self.d, label="output")
            
            
            nengo.Connection(stack_in, stack_out)
            nengo.Connection(self.sigin, stack_in[d])
            nengo.Connection(self.input, stack_in[:d])
            nengo.Connection(stack_out[d], self.sigout)
            nengo.Connection(stack_out[:d], self.output)

class SemanticNode(nengo.Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.type = spa.types.TScalar
        self.input = self

    def connect_to(self, other, **kwargs):
        return nengo.Connection(self, other, **kwargs)

class Dispatcher(spa.Network):
    def __init__(self, inp, circuits_dict, busy_node, label="dispatcher", vocab=voc):
        super().__init__(label=label)
        
        self.circuits_dict = circuits_dict
        
        with self:
            self.input = SemanticNode(size_in=vocab.dimensions)
            
            in_reg = spa.State(vocab)
            nengo.Connection(self.input, in_reg.input, label="in_reg")

            #go = spa.SemanticPointer([1], name="S_GO")
            go = SemanticNode([1], label="GO!")
            wait = SemanticNode(size_in=1, label="wait")
    
            #print(list(circuits_dict.items())[:4])

            switch = spa.ActionSelection()
            with switch:
                spa.ifmax(theta, RoutedConnection(go, wait))
                for keyword, circuit in circuits_dict.items():
                    spa.ifmax(
                            in_reg @ vocab[keyword],
                            RoutedConnection(go, circuit),
                            )
        
        for keyword, circuit in circuits_dict.items():
            nengo.Connection(circuit.input, busy_node[0])
            nengo.Connection(circuit.output, busy_node[1])

        #print(list(circuits_dict.items())[:4])

def create_modification_node(vocab, circuits, theta=0.2):
    d = vocab.dimensions
    pop_vec = vocab["S_POP"].v
    push_vec = vocab["S_PUSH"].v
    circ_holo = sum(vocab[k].v for k in circuits.keys())

    def modify_output(t, x):
        output_vec = x[:d]
        resume = x[-1] < -theta
        
        norm_output = np.linalg.norm(output_vec)
        norm_pop = np.linalg.norm(pop_vec)
        cos_sim = 0
        if norm_output > 1e-6 and norm_pop > 1e-6:
            cos_sim = np.dot(output_vec, pop_vec) / (norm_output * norm_pop)
        
        is_word = (output_vec @ circ_holo) > theta
        to_stack = vocab['Zero'].v
        to_dispatcher = vocab['Zero'].v
        
        if output_vec @ output_vec < theta:
            pass
        elif is_word:
            to_dispatcher = output_vec
        elif cos_sim > theta:
            to_stack = pop_vec
        else:
            combined = push_vec + output_vec
            norm_combined = np.linalg.norm(combined)
            if norm_combined > 1e-6:
                combined /= norm_combined
            to_stack = combined
        return np.concatenate((to_stack, to_dispatcher, [is_word and not resume]))
    
    return modify_output

def make_busy_signal():
    state = 0
    def busy_signal(t, sig):
        nonlocal state
        go = sig[0]
        done = sig[1]

        if not state and go > theta and not done > theta:
            state = 1
        if state and done > theta:
            state = 0

        return state
    return busy_signal

class WordCircuit(spa.Network):
    def __init__(self,  vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = spa.types.TScalar
        with self:
            self.input = nengo.Node(size_in=1, 
                                    size_out=1, 
                                    output=lambda _, x: x if x[0] > theta else 0, 
                                    label="input",
                                    )
            self.output = nengo.Node(size_in=1, label="output")

class SwapCircuit(WordCircuit):
    """(a b -- b a)
    """
    def __init__(self, stack, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.stack = stack

        if isinstance(stack, list):
            with self:
                def make_swap(stack):
                    state = 0
                    stopwatch = 0
                    def swap(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            stack[-2:] = stack[:-3:-1]
                            stopwatch = t
                            print([p.name for p in stack])
                        elif state == 0 and go > theta and t > stopwatch + 1.25:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + 1.75:
                            state = 0
                            stopwatch = 0
                        return state
                    return swap
                swapper = nengo.Node(size_in=1, output=make_swap(self.stack))

                nengo.Connection(self.input, swapper)
                nengo.Connection(swapper, self.output)


        elif isinstance(stack, nengo.Network):
            raise NotImplementedError("I'll do this later lol")

class UserFuncCircuit(WordCircuit):
    def __init__(self, func_register, func_controller, keys={}, bindings={}, vocab=voc, *args, **kwargs):
        super().__init__(*args, *kwargs)

        d = vocab.dimensions
        self.keys = keys # dictionary associating keywords to vector-encoded forth words
        self.bindings = bindings # dictionary associating keywords to function defs
        
        def make_prog_table(keys, bindings):
            state = 0
            stopwatch = 0
            ctrl_sig = 0
            def prog_table(t, x):
                nonlocal state, stopwatch, ctrl_sig
                func = x[:d]
                go = x[-1]
                to_controller = np.zeros(d)
                if state == 0 and go > theta and stopwatch == 0:
                    keys_list = list(keys.items())
                    key = keys_list[np.argmax([func @ k for _, k in keys_list])]
                    function = bindings[key]
                    to_controller = function
                    ctrl_sig = 1
                    stopwatch = t
                elif state == 0 and go > theta and t > stopwatch + 0.25:
                    ctrl_sig = 0
                elif state == 0 and go > theta and t > stopwatch + 1.25:
                    state = 1
                elif state == 1 and go < theta and t > stopwatch + 1.75:
                    state = 0
                    stopwatch = 0
                return np.concatenate([to_controller, ctrl_sig, state])
            return prog_table
        program_table = nengo.Node(size_in=1, output=make_prog_table(self.keys, self.bindings))
        nengo.Connection(program_table[:d], func_controller.input)
        nengo.Connection(program_table[-2], func_controller.sigin)

        nengo.Connection(self.input, program_table[-1])
        nengo.Connection(program_table[-1], self.output)

class RegisterBank(spa.Network):
    def __init__(self, names, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.bindings = {}

        with self:
            for name in names:
                setattr(self, name, spa.State(vocab))
                self.bindings[name] = getattr(self, name)


model = spa.Network()
with model:
    kws = """Keywords:
        :           F_FUNC
        ;           F_END
        >r          F_PUSHRET
        swap        F_SWAP
        @           F_PEEP
        rot         F_ROT
        -           F_SUB
        dup         F_DUP
        !           F_PUT
        r>          F_POPRET
        0<          F_ISNEG
        if          F_IF
        else        F_ELSE
        drop        F_DROP
        then        F_THEN
        execute     F_EXEC
    """

    data_stack = SimpleStack(label="data_stack")
    return_stack = SimpleStack(label="return_stack")
    call_stack = SimpleStack(label="call_stack")

    registers = RegisterBank(
            ['R1', 'R2', 'R3', 'R4', 'R5', 
             'I1', 'I2', 'I3', 
             'O1', 'O2', 'O3']
            )

    def new_dummy(name=""):
        dummy_circuit = spa.Network(f"{name} circuit")
        with dummy_circuit:
            dummy_circuit.sigin = nengo.Node(size_in=1, 
                                            size_out=1, 
                                            output=lambda _, x: x if x[0] > theta else 0, 
                                            label="sigin",
                                            )
            dummy_circuit.sigout = nengo.Node(size_in=1, label="sigout")
            dummy_circuit.type = spa.types.TScalar
            dummy_circuit.input = dummy_circuit.sigin
            dummy_circuit.output = dummy_circuit.sigout

        return dummy_circuit
   
    wds_circuits = spa.Network()
    with wds_circuits:
        circuits_dict = {"F_FUNC":      new_dummy("F_FUNC"),
                         "F_END":       new_dummy("F_END"),
                         "F_PUSHRET":   new_dummy("F_PUSHRET"),
                         "F_SWAP":      SwapCircuit(data_stack.stack, vocab=voc, label="SWAP Circuit"),
                         'F_PEEP':      new_dummy("F_PEEP"), 
                         'F_ROT':       new_dummy("F_ROT"),  
                         'F_SUB':       new_dummy("F_SUB"), 
                         'F_DUP':       new_dummy("F_DUP"),  
                         'F_PUT':       new_dummy("F_PUT"),  
                         'F_POPRET':    new_dummy("F_POPRET"),  
                         'F_ISNEG':     new_dummy("F_ISNEG"),
                         'F_IF':        new_dummy("F_IF"),
                         'F_ELSE':      new_dummy("F_ELSE"),  
                         'F_DROP':      new_dummy("F_DROP"),
                         'F_THEN':      new_dummy("F_THEN"),
                         'F_EXEC':      new_dummy("F_EXEC"),
                         }
    
    voc_items = ["R_LEFT", "R_RIGHT", "R_PHI", "T_NIL",
                 "APPLE", "BANANA", "CHERRY",
                 "S_PUSH", "S_POP", "S_DUMP", "S_CODE_ERR_STACKEMPTY", 'S_WORD',
                 ] + list(circuits_dict.keys())
    voc.populate("; ".join(voc_items))

    # holo = sum([voc[c].v for c in circuits_dict.keys()])
    # print(np.sqrt(len(circuits_dict.values())), 
    #       np.linalg.norm(holo),
    #      [voc[c].v @ holo for c in circuits_dict.keys()])
    
    lis = cons(voc['CHERRY'], cons(voc["APPLE"], voc["BANANA"]))
    listail1 = cons(voc["APPLE"], voc["BANANA"])
    listail2 = cons(voc["BANANA"],voc["T_NIL"])
    
    test_program = make_list(["APPLE", "CHERRY", "F_SWAP", "BANANA"], vocab=voc)
    
    voc.add(test_program.name, test_program.v)
    
    inp = spa.State(voc)
    out = spa.State(voc)
    
    
    nengo.Connection(inp.output, data_stack.input)
    nengo.Connection(data_stack.output, out.input)

    control_unit = ControlUnit(d=d, theta=theta, items=test_program, label="ControlUnit Network", voc=voc, assoc_memory=assoc, circuits=circuits_dict)

    
    nengo.Connection(control_unit.to_data_stack, inp.input)
    
    nengo.Connection(control_unit.trigger, data_stack.sigin)
    
    
    busy_node = nengo.Node(output=make_busy_signal(), size_in=2, size_out=1, label="busy_node")
    nengo.Connection(busy_node, control_unit.word_busy)

    dispatcher = Dispatcher(inp, circuits_dict, busy_node, vocab=voc)

    nengo.Connection(control_unit.to_dispatcher, dispatcher.input)
