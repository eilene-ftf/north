import nengo
import nengo_spa as spa
from nengo_spa.connectors import RoutedConnection

import numpy as np
from numpy.linalg import norm
from collections import UserDict

theta = 0.3     # threshold parameter
stability_threshold = 0.7 # thresholds stability of R_state
d = 256         # dimensionality
t_resume = 0.1 # do not make relative
t_delay = 0.5 # delay for func_ctrl to wait before populating tail
t_stack = 0.2
t_buffer = 0.25 # extra time padding so the function controller doesn't race the function return
t_ctrl = 0.25
t_busy = 0.5
t_done = 0.75
clock_tick = 2.0

assert t_ctrl < t_busy and t_busy < t_done

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
        if new_name not in vocab:
            vocab.add(new_name, spa.SemanticPointer(new_sp.normalized().v, name=new_name))
        return vocab[new_name]
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
        
#   def make_list(lis, vocab=voc):
#
#       l = vocab[lis.pop()]
#       #print(l)
#       while lis:
#           l = cons(vocab[lis.pop()], l)
#       return l
    
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

class FallingEdgeDetector(nengo.Network):
    def __init__(self, tau=0.01, n_neurons=100, bias=-0.5, **kwargs):
        super().__init__(**kwargs)
        
        with self:
            self.input = nengo.Node(size_in=1)
            
            self.delayed = nengo.Node(size_in=1)
            nengo.Connection(self.input, self.delayed, synapse=tau)
            
            self.compare = nengo.Ensemble(n_neurons, 1)
            
            nengo.Connection(self.input, self.compare, transform=-2)
            nengo.Connection(self.delayed, self.compare, transform=2)
            nengo.Connection(nengo.Node(output=bias, label="bias"), self.compare)
            
            self.output = nengo.Node(size_in=1)
            nengo.Connection(self.compare, self.output)

def dereference(t, x):
    closest_sp = voc._keys[np.argmax(voc.dot(x))]
    if closest_sp.startswith("Pointer_to_"):
        return voc[closest_sp[11:]].v
    return cleanup(spa.SemanticPointer(x), vocab=voc).v            
            
class ControlUnit(spa.Network):
    def __init__(self, circuits, d=d, theta=theta, items=None, label="ControlUnit", vocab=spa.Vocabulary(d), assoc_memory=assoc):
        super().__init__(label=label)
        self.d = d
        self.theta = theta
        self.vocab = vocab
        to_add = [l for l in ['R_LEFT', 'R_RIGHT', 'R_PHI', 'T_NIL'] if l not in vocab]
        vocab.populate(';'.join(to_add))
        self.assoc_memory = assoc_memory
        
        self.vector_numbers = items.v
            
        with self:
            # Labeled states
            R_state = spa.State(self.vocab, label="R_register")
            H_state = spa.State(self.vocab, label="head_register")
            T_state = spa.State(self.vocab, label="tail_register")
            output_state = spa.State(self.vocab, label="output_register")
            
            self.clock_trigger = nengo.Node(
                lambda t: 1.0 if (t % clock_tick) < 0.1 else 0.0, 
                label="clock_trigger"
            )
            
            edge_detector = RisingEdgeDetector(tau=0.04, bias=0, label="edge_detector")
            nengo.Connection(self.clock_trigger, edge_detector.input)
            
            mod_node = nengo.Node(
                create_modification_node(vocab, circuits=circuits, theta=theta),
                size_in=2*d+2,
                size_out=2*d+1,
                label="mod_node"
            )
            
            self.output = mod_node

            self.holo_node = nengo.Node(size_in=d, size_out=d, label="user_func_holo")
            nengo.Connection(self.holo_node, mod_node[d:d*2])


            control_check = nengo.Node(
                    create_control_node(voc),
                    size_in=2*voc.dimensions + 1, 
                    size_out=2*voc.dimensions + 1, 
                    label="control_check")

            self.flag_control = control_check
           
            nengo.Connection(output_state.output, control_check[:d])
            nengo.Connection(control_check[:d], mod_node[:d])
            nengo.Connection(control_check[-1], control_check[-1])

            self.is_paused = nengo.Node(size_in=1, label="paused?")
            nengo.Connection(mod_node[-1], self.is_paused)

            self.to_data_stack = nengo.Node(size_in=d, label="to_data_stack")
            nengo.Connection(mod_node[:d], self.to_data_stack)

            self.to_dispatcher = nengo.Node(size_in=d, label="to_dispatcher")
            nengo.Connection(mod_node[d:d*2], self.to_dispatcher)
            
            
            self.input_node = nengo.Node(output=self.vector_numbers, label="input_node")
            self.word_busy = nengo.Node(size_in=1, label="word_busy")
            self.condition_output = nengo.Node(size_in=1, label="condition_flag")
            
            def make_resume_state():
                state = 0
                stopwatch = 0
                def resume_state_machine(t, x):
                    nonlocal state, stopwatch
                    resume_edge = x[0]
                    clock = x[1]

                    if state == 0 and resume_edge > theta:
                        state = 1
                        #print(f"Detected resume signal, entering state {state}")
                    elif state == 1 and clock > 1-theta and stopwatch == 0: 
                        stopwatch = t
                        #print("Clock pulse arrived, waiting to reset")
                    elif state == 1 and stopwatch != 0 and t > stopwatch + t_resume:
                        state = 0                        
                        stopwatch = 0
                        #print(f"Resetting, entering state {state}")

                    #print(f'state:{state} stopwatch:{stopwatch:.2f} time:{t:.2f} clock:{clock:.2f} resume:{resume_edge:.2f}')
                    return state
                return resume_state_machine

            # May need to be changed, seems like it might behave weirdly if 
            # there are multiple words in a row or processing a word takes a 
            # long time
            resume = FallingEdgeDetector(bias=0, label="resume")
            resume_state = nengo.Node(size_in=2,
                                      size_out=1,
                                      output=make_resume_state(),
                                      label="resume_state"
                                      )
            nengo.Connection(self.word_busy, resume.input)
            nengo.Connection(resume.output, resume_state[0])
            nengo.Connection(self.clock_trigger, resume_state[1])
            nengo.Connection(resume_state, mod_node[-1])
            
            def cleanup_node(t, x, vocab=self.vocab):
                return cleanup(x, vocab=vocab).v
            
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

            noisy_h = spa.State(self.vocab, label="noisy_head")
            noisy_t = spa.State(self.vocab, label="noisy_tail")
            
            (~self.vocab['R_LEFT'] * R_state) >> noisy_h
            (~self.vocab['R_RIGHT'] * R_state) >> noisy_t
            
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
            #nengo.Connection(tail_cleanup, T_state.input)
            
            def compute_condition(t, x):
                R_vec = x[0:self.d]
                edge_detected = x[self.d]
                pause = x[self.d + 1]
                dot_val = np.dot(R_vec, self.vocab['T_NIL'].v)
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
            
            threshold_node = nengo.Node(
                size_in=1,
                size_out=1,
                output=lambda t, x: 1 if x < stability_threshold else -1,
                label="threshold_node"
            )
            nengo.Connection(stable, threshold_node)
            nengo.Connection(threshold_node, mod_node[-2])
            
            def make_latch_node():
                state = 0
                def latch_func(t, x):
                    nonlocal state
                    condition = x[0]
                    stable = x[1]
                    if stable < -0.5:
                        state = 0
                    if condition > 0.5:
                        state = 1
                    return [state]
                return latch_func
            
            latch_node = nengo.Node(
                make_latch_node(), 
                size_in=2, 
                size_out=1,
                label="latch_node"
            )
            nengo.Connection(condition_node, latch_node[0])
            nengo.Connection(threshold_node, latch_node[1])

            def t_state_gate(t, x):
                new_t = x[0:self.d]
                current_t = x[self.d:2*self.d]
                latch = x[-2]
                pause = x[-1]
                if latch > 0.5 or pause > theta:
                    return current_t
                else:
                    return new_t
            
            t_gate_node = nengo.Node(
                t_state_gate, 
                size_in=2*self.d+2, 
                size_out=self.d,
                label="t_gate_node"
            )
            nengo.Connection(tail_cleanup, t_gate_node[:self.d])
            nengo.Connection(T_state.output, t_gate_node[self.d:2*self.d])
            nengo.Connection(latch_node, t_gate_node[-2])
            nengo.Connection(mod_node[-1], t_gate_node[-1])
            
            nengo.Connection(H_state.output, output_state.input)
            
            def update_R(t, x):
                T_vec = x[0:self.d]
                R_vec = x[self.d:2*self.d]
                cond = x[2*self.d]
                return cleanup(spa.SemanticPointer(T_vec if cond > 0.5 else R_vec), vocab=self.vocab).v
            
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
            
            def make_func_rtrn(vocab=voc):
                d = vocab.dimensions
                state = 0
                stopwatch = 0
                stack_cmd = np.zeros(d)
                to_rstack = np.zeros(d)
                to_return = np.zeros(d)

                def function_return(t, x):
                    nonlocal state, stopwatch, to_return, stack_cmd, to_rstack 
                    
                    from_R_state = x[:d]
                    from_T_state = x[d:d*2]
                    from_call_stack = x[d*2:d*3]
                    stack_done_signal = x[-1]
                    clock = x[-2]
                    
                    is_nil = from_T_state @ vocab['T_NIL'].v > theta

                    if state == 0 and not is_nil:
                        to_return = from_R_state
                        to_rstack = np.zeros(d)
                    elif state == 0 and is_nil:
                        state = 1
                        stack_cmd = vocab['S_POP'].v
                        to_rstack = vocab['S_DUMP'].v
                    elif state == 1 and stack_done_signal > theta:
                        state = 2
                        stack_cmd = np.zeros(d)
                        if (from_call_stack @ vocab['S_CODE_ERR_STACKEMPTY'].v) < theta:
                            to_return = from_call_stack
                        else:
                            to_return = vocab['S_CODE_HALT'].v
                    elif state == 2 and clock > 1-theta:
                        stopwatch = t
                        to_rstack = np.zeros(d)
                        state = 3
                    elif state == 3 and t > stopwatch + t_resume:
                        state = 0
                    
                    #print(f'{t:.2f} {state} {from_T_state @ vocab["T_NIL"].v:.2f} {clock:.2f}')
                    return np.concatenate([to_return, stack_cmd, to_rstack, [state]])
                return function_return

            def make_func_ctrl(vocab=voc):
                d = vocab.dimensions
                state = 0
                stopwatch = 0
                to_return = np.zeros(d)
                ctrl_flag = False
                def function_controller(t, x):
                    nonlocal state, stopwatch, to_return, ctrl_flag
                    from_R_state = x[:d]
                    from_user_func = x[d:d*2]
                    latch = x[-4] # get rid of this, not necessary
                    clock = x[-3]
                    ctrl_signal = x[-1]
                    stack_done_signal = x[-2]

                    pushmag = (from_R_state @ vocab['S_PUSH'].v)
                    if pushmag > theta:
                        from_R_state -= vocab['S_PUSH'].v

                    if ctrl_signal > theta and not ctrl_flag:
                        ctrl_flag = True

                    if not ctrl_flag and state == 0:
                        to_return = from_R_state 
                    elif ctrl_flag and state == 0 and latch > theta:
                        to_return = from_R_state
                    elif ctrl_flag and state == 0 and latch < theta: #and stopwatch == 0:
                        #stopwatch = t
                    #elif ctrl_flag and state == 0 and latch < theta and t > stopwatch + t_delay:
                        state = 1 # push state
                        to_return = from_R_state + vocab['S_PUSH'].v 
                        print(f"at {t:.2f}: pushing tail, entered state {state}")
                    elif state == 1 and stack_done_signal > theta:
                        stopwatch = t
                        state = 2 # move retrieved function to T_state
                        to_return = from_user_func
                        print(f"at {t:.2f}: putting userfunc in tail, entered state {state}")
                    elif state == 2 and t > stopwatch + t_ctrl:
                        stopwatch = 0
                        state = 3 # done, wait for reset
                        to_return = from_user_func
                        print(f"at {t:.2f}: ready to reset, entered state {state}")
                    elif state == 3 and clock > 1-theta and stopwatch == 0:
                        stopwatch = t
                        print(f"at {t:.2f}: clock pulse received, starting timer")
                    elif state == 3 and stopwatch != 0 and t > stopwatch + t_resume:
                        state = 0 # reset
                        stopwatch = 0
                        ctrl_flag = False
                        print(f"at {t:.2f}: reset, entered state {state}")
                    
                    #print(ctrl_flag, latch < theta)

                    return np.concatenate([to_return, [state]])

                return function_controller
            
            # ALERT: THERE MAY BE A RACE CONDITION ON THE CALL STACK

            self.from_call_stack = nengo.Node(size_in=d)
            func_rtrn = nengo.Node(size_in=d*3+2,
                                   output=make_func_rtrn(vocab=vocab),
                                   label="func_rtrn")

            self.func_ctrl_sigin = nengo.Node(size_in=1)
            self.call_stack_sigin = nengo.Node(size_in=1)
            self.from_user_func = nengo.Node(size_in=d)
            self.func_ctrl = nengo.Node(size_in=d*2+4, 
                                        output=make_func_ctrl(vocab=vocab), 
                                        label="func_ctrl")
            #func_ctrl_state = nengo.Node(size_in=1, label="func_ctrl_state")
            #nengo.Connection(self.func_ctrl[-1], func_ctrl_state)
            nengo.Connection(self.call_stack_sigin, func_rtrn[-1])
            nengo.Connection(T_state.output, func_rtrn[d:d*2])
            nengo.Connection(self.from_call_stack, func_rtrn[d*2:d*3])
            nengo.Connection(self.func_ctrl_sigin, self.func_ctrl[-1]) 
            nengo.Connection(self.call_stack_sigin, self.func_ctrl[-2])
            nengo.Connection(self.clock_trigger, func_rtrn[-2])
            nengo.Connection(self.clock_trigger, self.func_ctrl[-3])
            nengo.Connection(threshold_node, self.func_ctrl[-4])
            nengo.Connection(t_gate_node, func_rtrn[:d], synapse=0.01)
            nengo.Connection(func_rtrn[:d], self.func_ctrl[:d])
            nengo.Connection(self.from_user_func, self.func_ctrl[d:d*2])

            nengo.Connection(self.func_ctrl[:d], T_state.input) 

            self.to_call_stack = nengo.Node(size_in=d)
            self.call_stack_sigout = nengo.Node(size_in=1, 
                                                size_out=1, 
                                                output=lambda t, x: [x[0] > 1-theta and x[0] < 1+theta]
                                                )
            nengo.Connection(T_state.output, self.to_call_stack)
            nengo.Connection(func_rtrn[d:d*2], self.to_call_stack)
            nengo.Connection(self.func_ctrl[-1], self.call_stack_sigout)

            self.func_ctrl_done = nengo.Node(size_in=1,
                                                 size_out=1,
                                                 output=lambda t, x: [x[0] > 3-theta and x[0] < 3+theta]
                                                 )

            nengo.Connection(func_rtrn[-1], self.call_stack_sigout)
            nengo.Connection(self.func_ctrl[-1], self.func_ctrl_done)

            self.to_return_stack = nengo.Node(size_in=d)
            nengo.Connection(func_rtrn[d*2:d*3], self.to_return_stack)

            test_T_flow = spa.State(vocab, label="test")
            test_T_2 = spa.State(vocab, label="test2")
            nengo.Connection(tail_cleanup, test_T_flow.input)
            nengo.Connection(t_gate_node[:d], test_T_2.input)
            
def make_stack_in(stack, tag='stack'):
    stopwatch = 0
    state = 0
    out = np.zeros(d)
    
    def stack_in(t, x, vocab=voc):
        nonlocal stopwatch, state, out
        if t < 0.1 and stack:
            del stack[:]
            print(f'{tag} state: {[p.name for p in stack]}')
        sig = x[d]
        inp = spa.SemanticPointer(x[:d])
        #if tag == "call_stack": print(sig, state)
        if state == 0 and sig > 1-theta:
            state = 1
            stopwatch = t
            if vcos(inp, vocab['S_PUSH']) > theta:
                stack.append(cleanup(inp - vocab['S_PUSH']))
                print(f'{tag} state: {[p.name for p in stack]}')
            elif vcos(inp, vocab['S_PEEK']) > theta:
                out = vocab['S_PEEK'].v
                print(f'{tag} state: {[p.name for p in stack]}')
            elif vcos(inp, vocab['S_POP']) > theta:
                out = vocab['S_POP'].v
                print(f'{tag} state: {[p.name for p in stack]}')
            elif vcos(inp, vocab['S_DUMP']) > theta:
                del stack[:]
                print(f'{tag} state: {[p.name for p in stack]}')
        if state == 1 and sig < theta and t > stopwatch + t_stack:
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
            if vcos(inp, vocab['S_PEEK']) > theta:
                print("peek!")
                if stack:
                    state = stack[-1].v
                else:
                    state = vocab['S_CODE_ERR_STACKEMPTY'].v
            elif vcos(inp, vocab['S_POP']) > theta:
                print("pop!")
                if stack:
                    state = stack.pop().v
                else:
                    state = vocab['S_CODE_ERR_STACKEMPTY'].v
                print([p.name for p in stack])
        if sigout == 1 and t > stopwatch + t_stack:
            sigout = 0
        return np.concatenate((state, [sigout]))
    
    return stack_out


class SimpleStack(spa.Network):
    def __init__(self, stack=None, d=d, label="stack memory"):
        super().__init__(label=label)
        self.d = d
        if stack is not None:
            self.stack = stack
        else:
            self.stack = []
        
        with self:
            stack_in = nengo.Node(size_in=self.d+1, 
                                  output=make_stack_in(self.stack, tag=self.label), 
                                  label="stack_in")
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
    def __init__(self, circuits, busy_node, label="dispatcher", vocab=voc):
        super().__init__(label=label)
        
        self.circuits_dict = circuits.circuits_dict
        d = vocab.dimensions
        
        with self:
            self.input = SemanticNode(size_in=d)
       
            in_reg = spa.State(vocab, label="in_reg")
            nengo.Connection(self.input, in_reg.input)

            self.holo_node = nengo.Node(size_in=d, label="holo node")
            holo_reg = spa.State(vocab, label="holo_reg")
            nengo.Connection(self.holo_node, holo_reg.input)

            holodot = SemanticNode(
                    size_in=d*2, 
                    size_out=1, 
                    output=lambda t, x: min(4 * (x[:d] @ x[d:]), 1), 
                    label="holodot"
                    )
            nengo.Connection(self.holo_node, holodot[:d])
            nengo.Connection(self.input, holodot[d:])

            hdot_ptr = spa.State(1, subdimensions=1)
            nengo.Connection(holodot, hdot_ptr.input)
            one = spa.SemanticPointer([1], name="one")

            #go = spa.SemanticPointer([1], name="S_GO")
            go = SemanticNode([1], label="GO!")
            wait = SemanticNode(size_in=1, label="wait")
    
            #print(list(circuits_dict.items())[:4])

            switch = spa.ActionSelection()
            with switch:
                spa.ifmax(theta, RoutedConnection(go, wait))
                for keyword, circuit in circuits.circuits_dict.items():
                    spa.ifmax(
                            in_reg @ vocab[keyword],
                            RoutedConnection(go, circuit),
                            )
                spa.ifmax(hdot_ptr @ one, RoutedConnection(go, circuits.user_func_circuit))
        
        for keyword, circuit in circuits.circuits_dict.items():
            nengo.Connection(circuit.input, busy_node[0])
            nengo.Connection(circuit.output, busy_node[1])

        nengo.Connection(circuits.user_func_circuit.input, busy_node[0])
        nengo.Connection(circuits.user_func_circuit.output, busy_node[1])

        #print(list(circuits_dict.items())[:4])
def create_control_node(vocab, theta=theta):
    d = vocab.dimensions
    output = []
    
    def flag_switcher(t, x):
        controlcode = spa.SemanticPointer(x[d:d*2])
        branch1 = vocab['S_ORIG_1']
        branch2 = vocab['S_ORIG_2']
        empty = vocab['S_CODE_ERR_STACKEMPTY']
        
        flag = x[-1]
        header = spa.SemanticPointer(x[:d])
        
        ifb = vocab['F_IF']
        elseb = vocab['F_ELSE']
        thenb = vocab['F_THEN']
        #want not the dot product, but the scalar projection of control code onto both branch codes

        if vcos(controlcode, empty) > theta:
            flag = 0
        if vcos(controlcode,branch1) > 1 - theta:
            #print("condition met")
            flag = 0
            if x[-1] > 1-theta:
                #print("still ignoring")
                flag = 1
            if vcos(header, elseb) > 1 - theta:
                #print("skip/ignore else branch")
                flag = 1
        if vcos(controlcode, branch2) > 1 - theta:
            #print("condition not met -> input is")
            flag = 1
            if vcos(header, elseb) > 1 - theta:
                #print("second branch starting")
                flag = 0
        if vcos(header, thenb) > 1 - theta:
            flag = 0
        return flag 

    def control_flow_state(t, x):
        nonlocal output
        header = x[:d]
        peep = vocab['S_PEEK'].v
        flag = flag_switcher(t,x)
        if flag == 0:
            output = np.concatenate((header, peep, [flag]))
        elif flag == 1:
            output = np.concatenate((np.zeros(len(header)), peep, [flag]))
        return output
    return control_flow_state 

def create_modification_node(vocab, circuits, theta=0.2):
    d = vocab.dimensions
    pop_vec = vocab["S_POP"].v
    push_vec = vocab["S_PUSH"].v
    circ_holo = sum(vocab[k].v for k in circuits.keys())
    R_flag = False
    stopwatch = 0

    def modify_output(t, x):
        nonlocal R_flag, stopwatch
        output_vec = x[:d]
        user_func_holo = x[d:d*2]
        R_is_stable = x[-2] < -1 + theta
        resume = x[-1] > 1-theta
        
        norm_output = np.linalg.norm(output_vec)
        norm_pop = np.linalg.norm(pop_vec)
        cos_sim = 0
        if norm_output > 1e-6 and norm_pop > 1e-6:
            cos_sim = np.dot(output_vec, pop_vec) / (norm_output * norm_pop)
        
        is_word = (output_vec @ (circ_holo + user_func_holo)) > theta
        to_stack = vocab['Zero'].v
        to_dispatcher = vocab['Zero'].v

        if (resume or t < 0.1) and R_flag:
            R_flag = False
            stopwatch = 0
        
        if output_vec @ output_vec < theta:
            pass
        elif is_word and R_is_stable and stopwatch == 0:
            stopwatch = t
        elif is_word and R_is_stable and t > stopwatch + t_delay:
            R_flag = True
            to_dispatcher = output_vec
        elif is_word:
            pass
        elif cos_sim > theta:
            to_stack = pop_vec
        else:
            combined = push_vec + output_vec
            norm_combined = np.linalg.norm(combined)
            if norm_combined > 1e-6:
                combined /= norm_combined
            to_stack = combined
        return np.concatenate((to_stack, to_dispatcher, [is_word and not resume and R_flag]))
    
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
                            if stack:
                                stack[-2:] = stack[:-3:-1]
                                stopwatch = t
                            print([p.name for p in stack])
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return swap
                swapper = nengo.Node(size_in=1, output=make_swap(self.stack))

                nengo.Connection(self.input, swapper)
                nengo.Connection(swapper, self.output)


        elif isinstance(stack, nengo.Network):
            raise NotImplementedError("I'll do this later lol")
            
            
class DupCircuit(WordCircuit):
    """(a -- a a)
    Duplicate top of stack
    """
    def __init__(self, stack, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stack = stack
        if isinstance (stack, list):
            with self:
                def make_duplicate(stack):
                    state = 0
                    stopwatch = 0
                    def duplicate(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if stack:
                                temp = stack.pop()
                                stack.append(temp)
                                stack.append(temp)
                                print([p.name for p in stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return duplicate 
                dupper = nengo.Node(size_in = 1, output = make_duplicate(stack))
                nengo.Connection(self.input, dupper)
                nengo.Connection(dupper, self.output)

class DropCircuit(WordCircuit):
    """(a -- )
    Remove top of stack
    """
    def __init__(self, stack, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack

        if isinstance(stack, list):
            with self:
                def make_drop(stack):
                    state = 0
                    stopwatch = 0
                    def drop(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if stack:
                                stack.pop()
                                print([p.name for p in stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return drop
                dropper = nengo.Node(size_in=1, output=make_drop(self.stack))
                nengo.Connection(self.input, dropper)
                nengo.Connection(dropper, self.output)

class AddCircuit(WordCircuit):
    """(a b -- c)
    Add two numbers using list-based arithmetic
    """
    def __init__(self, stack, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.vocab = vocab

        if isinstance(stack, list):
            with self:
                def make_add(stack, vocab):
                    state = 0
                    stopwatch = 0
                    def add(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if len(stack) >= 2:
                                b = stack.pop()  # Second operand (top)
                                a = stack.pop()  # First operand
                                # Addition: concatenate lists representing numbers (we utilize Peano based encoding, reference add_list_numbers())
                                # a = (NIL) = 1, b = ((NIL)) = 2, result = (((NIL))) = 3 ; We know this is terrible, its all we had time for (final will use modular encoding)
                                result = add_list_numbers(a, b, vocab)
                                stack.append(result)
                                print([p.name for p in stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return add
                adder = nengo.Node(size_in=1, output=make_add(self.stack, self.vocab))
                nengo.Connection(self.input, adder)
                nengo.Connection(adder, self.output)

class SubCircuit(WordCircuit):
    """(a b -- c)
    Subtract two numbers using list-based arithmetic
    """
    def __init__(self, stack, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.vocab = vocab

        if isinstance(stack, list):
            with self:
                def make_sub(stack, vocab):
                    state = 0
                    stopwatch = 0
                    def sub(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if len(stack) >= 2:
                                b = stack.pop()  # Second operand (top)
                                a = stack.pop()  # First operand
                                # Subtraction: reduce list depth
                                result = sub_list_numbers(a, b, vocab)
                                stack.append(result)
                                print([p.name for p in stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return sub
                subtractor = nengo.Node(size_in=1, output=make_sub(self.stack, self.vocab))
                nengo.Connection(self.input, subtractor)
                nengo.Connection(subtractor, self.output)

class IsZeroCircuit(WordCircuit):
    """(a -- flag)
    Test if top of stack equals zero (NIL)
    """
    def __init__(self, stack, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.vocab = vocab

        if isinstance(stack, list):
            with self:
                def make_is_zero(stack, vocab):
                    state = 0
                    stopwatch = 0
                    def is_zero(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if stack:
                                a = stack.pop()
                                # Test if a equals NIL (zero)
                                is_nil = vcos(a, vocab['T_NIL']) > theta
                                flag = vocab['TRUE'] if is_nil else vocab['FALSE']
                                stack.append(flag)
                                print([p.name for p in stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return is_zero
                tester = nengo.Node(size_in=1, output=make_is_zero(self.stack, self.vocab))
                nengo.Connection(self.input, tester)
                nengo.Connection(tester, self.output)

class PeepCircuit(WordCircuit):
    """@ (addr -- value)
    Fetch from memory address
    """
    def __init__(self, stack, memory_registers, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.memory_registers = memory_registers
        self.vocab = vocab

        if isinstance(stack, list):
            with self:
                def make_peep(stack, registers, vocab):
                    state = 0
                    stopwatch = 0
                    def peep(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if stack:
                                addr = stack.pop()
                                # Use cleanup to find closest register address
                                reg_name = cleanup(addr, vocab).name
                                if reg_name in registers.bindings:
                                    value = registers.bindings[reg_name].output
                                    # Convert nengo output to semantic pointer
                                    value_sp = spa.SemanticPointer(value)
                                    stack.append(value_sp)
                                else:
                                    # Default to NIL if address not found
                                    stack.append(vocab['T_NIL'])
                                print([p.name for p in stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return peep
                peeper = nengo.Node(size_in=1, output=make_peep(self.stack, self.memory_registers, self.vocab))
                nengo.Connection(self.input, peeper)
                nengo.Connection(peeper, self.output)

class PutCircuit(WordCircuit):
    """! (value addr -- )
    Store value at memory address
    """
    def __init__(self, stack, memory_registers, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.memory_registers = memory_registers
        self.vocab = vocab

        if isinstance(stack, list):
            with self:
                def make_put(stack, registers, vocab):
                    state = 0
                    stopwatch = 0
                    def put(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if len(stack) >= 2:
                                addr = stack.pop()  # Address (top)
                                value = stack.pop()  # Value
                                # Use cleanup to find closest register address
                                reg_name = cleanup(addr, vocab).name
                                if reg_name in registers.bindings:
                                    # Store value in register (this is simplified)
                                    # In practice, you'd need to handle nengo connections
                                    registers.bindings[reg_name].input = value.v
                                print([p.name for p in stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return put
                putter = nengo.Node(size_in=1, output=make_put(self.stack, self.memory_registers, self.vocab))
                nengo.Connection(self.input, putter)
                nengo.Connection(putter, self.output)

class FuncCircuit(WordCircuit):
    """: Start word definition
    Begins compilation mode
    """
    def __init__(self, compilation_state, dictionary, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compilation_state = compilation_state
        self.dictionary = dictionary
        self.vocab = vocab

        if isinstance(compilation_state, list):
            with self:
                def make_func(comp_state, dictionary, vocab):
                    state = 0
                    stopwatch = 0
                    def func(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            # Enter compilation mode
                            comp_state.append(vocab['COMPILING'])
                            # Start new word definition
                            dictionary.append([])  # New empty definition
                            print("Starting word definition")
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return func
                func_starter = nengo.Node(size_in=1, output=make_func(self.compilation_state, self.dictionary, self.vocab))
                nengo.Connection(self.input, func_starter)
                nengo.Connection(func_starter, self.output)

class EndCircuit(WordCircuit):
    """; End word definition
    Ends compilation mode and creates word
    """
    def __init__(self, compilation_state, dictionary, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compilation_state = compilation_state
        self.dictionary = dictionary
        self.vocab = vocab

        if isinstance(compilation_state, list):
            with self:
                def make_end(comp_state, dictionary, vocab):
                    state = 0
                    stopwatch = 0
                    def end(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if comp_state and dictionary:
                                # Exit compilation mode
                                comp_state.pop()
                                # Finalize word definition
                                word_def = dictionary.pop()
                                # Store in vocabulary (simplified)
                                print(f"Ending word definition: {[w.name for w in word_def]}")
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return end
                end_definer = nengo.Node(size_in=1, output=make_end(self.compilation_state, self.dictionary, self.vocab))
                nengo.Connection(self.input, end_definer)
                nengo.Connection(end_definer, self.output)


class IfCircuit(WordCircuit):
    """IF conditional execution
    Pops flag, pushes control flow state


    """
    def __init__(self, stack, ctrl_flow_stack, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = stack
        self.ctrl_flow_stack = ctrl_flow_stack
        self.vocab = vocab

        if isinstance(stack, list) and isinstance(ctrl_flow_stack, list):
            with self:
                def make_if(stack, ctrl_flow_stack, vocab):
                    state = 0
                    stopwatch = 0
                    def if_stmt(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if stack:
                                flag = stack.pop()
                                # Check if condition is true
                                is_true = vcos(flag, vocab['TRUE']) > theta
                                # Push execution state to control flow stack
                                ctrl_flow_stack.append(vocab['S_ORIG_1'] if is_true else vocab['S_ORIG_2'])
                                print(f"IF: condition is {'true' if is_true else 'false'}")
                                print([p.name for p in ctrl_flow_stack])
                            stopwatch = t
                            state = 0
                        elif state == 0 and go > theta and t > stopwatch + 1.25:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + 1.75:
                            state = 0
                            stopwatch = 0
                        return state
                    return if_stmt
                if_processor = nengo.Node(size_in=1, output=make_if(self.stack, self.ctrl_flow_stack, self.vocab))
                nengo.Connection(self.input, if_processor)
                nengo.Connection(if_processor, self.output)

class ElseCircuit(WordCircuit):
    """THEN end conditional
    Pops control flow state
    """
    def __init__(self, ctrl_flow_stack, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl_flow_stack = ctrl_flow_stack
        self.vocab = vocab

        if isinstance(ctrl_flow_stack, list):
            with self:
                def make_else(ctrl_flow_stack, vocab):
                    state = 0
                    stopwatch = 0
                    def then_stmt(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if ctrl_flow_stack:
                                # Pop control flow state
                                condition = ctrl_flow_stack.pop()
                                print(f"ELSE: running branch when condition false")
                                print([p.name for p in ctrl_flow_stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + 1.25:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + 1.75:
                            state = 0
                            stopwatch = 0
                        return state
                    return then_stmt
                else_processor = nengo.Node(size_in=1, output=make_else(self.ctrl_flow_stack, self.vocab))
                nengo.Connection(self.input, else_processor)
                nengo.Connection(else_processor, self.output)

class ThenCircuit(WordCircuit):
    """THEN end conditional
    Pops control flow state
    """
    def __init__(self, ctrl_flow_stack, vocab=voc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctrl_flow_stack = ctrl_flow_stack
        self.vocab = vocab

        if isinstance(ctrl_flow_stack, list):
            with self:
                def make_then(ctrl_flow_stack, vocab):
                    state = 0
                    stopwatch = 0
                    def then_stmt(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            if ctrl_flow_stack:
                                # Pop control flow state
                                condition = ctrl_flow_stack.pop()
                                print(f"THEN: ending conditional block")
                                print([p.name for p in ctrl_flow_stack])
                            stopwatch = t
                        elif state == 0 and go > theta and t > stopwatch + 1.25:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + 1.75:
                            state = 0
                            stopwatch = 0
                        return state
                    return then_stmt
                then_processor = nengo.Node(size_in=1, output=make_then(self.ctrl_flow_stack, self.vocab))
                nengo.Connection(self.input, then_processor)
                nengo.Connection(then_processor, self.output)

# Helper functions for list-based arithmetic
def add_list_numbers(a, b, vocab):
    """Add two list-encoded numbers"""
    # Count depth of nested lists for a and b because for some reason, we utilize the fucking Peano construction
    depth_a = count_list_depth(a, vocab)
    #depth_b = count_list_depth(b, vocab)
    
    # Create result with depth = depth_a + depth_b again, because God cursed us with Peano... AAAGHHHHHHHHH
    result = b  # Start with b
    for i in range(depth_a):
        result = cons(vocab['T_NIL'], result, vocab=vocab)
    
    return result

def sub_list_numbers(a, b, vocab):
    """Subtract two list-encoded numbers"""
    depth_a = count_list_depth(a, vocab)
    depth_b = count_list_depth(b, vocab)
    
    # Result depth = max(0, depth_a - depth_b)
    result_depth = max(0, depth_a - depth_b)
    
    result = vocab['T_NIL']
    for i in range(result_depth):
        result = cons(result, vocab['T_NIL'])
    
    return result

def count_list_depth(lst, vocab):
    """Count nesting depth of list (NIL = 0, (NIL) = 1, etc.) iteratively."""
    depth = 0
    current = lst
    while vcos(current, vocab['T_NIL']) < theta:
        depth += 1
        current = car(current, vocab)
    return depth

class DropCircuit(WordCircuit):
    def __init__(self, stack, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stack = stack
        if isinstance (stack, list):
            with self:
                def make_drop(stack):
                    state = 0
                    stopwatch = 0
                    def drop(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            stack.pop().v 
                            stopwatch = t
                            print([p.name for p in stack])
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return drop 
                dropper = nengo.Node(size_in = 1, output = make_drop(stack))
                nengo.Connection(self.input, dropper)
                nengo.Connection(dropper, self.output)

class DupCircuit(WordCircuit):
    def __init__(self, stack, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.stack = stack
        if isinstance (stack, list):
            with self:
                def make_duplicate(stack):
                    state = 0
                    stopwatch = 0
                    def duplicate(t, x):
                        nonlocal state, stopwatch
                        go = x[0]
                        if state == 0 and go > theta and stopwatch == 0:
                            temp = stack.pop().v
                            stack.append(temp)
                            stack.append(temp)
                            stopwatch = t
                            print([p.name for p in stack])
                        elif state == 0 and go > theta and t > stopwatch + t_busy:
                            state = 1
                        elif state == 1 and go < theta and t > stopwatch + t_done:
                            state = 0
                            stopwatch = 0
                        return state
                    return duplicate 
                dupper = nengo.Node(size_in = 1, output = make_duplicate(stack))
                nengo.Connection(self.input, dupper)
                nengo.Connection(dupper, self.output)

def get_or_add_and_get(term, vocab, v=None):
    if term not in vocab:
        if v is None: v = vocab.create_pointer()
        vocab.add(term, v)
    return vocab[term]

class UserFuncCircuit(WordCircuit):
    def __init__(self, user_table={}, vocab=voc, *args, **kwargs):
        super().__init__(*args, *kwargs)

        d = vocab.dimensions
        self.table = user_table
        # dictiinary associating keywords to vector-encoded forth words
        self.words = {k: v for k, v in zip(
                        user_table.keys(), 
                        (get_or_add_and_get(t, vocab) for t in user_table.keys())
                        )
                     } 
        self.bindings = {k: v for k, v in zip(
                    user_table.keys(),
                    (make_list(r, vocab=vocab) for r in user_table.values()))
                        } # dictionary associating keywords to function defs

        self.holo = sum(p.v for p in self.words.values())
        
        def make_prog_table(words, bindings):
            state = 0
            stopwatch = 0
            ctrl_sig = 0
            to_controller = np.zeros(d)
            def prog_table(t, x):
                nonlocal state, stopwatch, ctrl_sig, to_controller
                func = x[:d]
                go = x[-1]
                if t < 0.01 and stopwatch != 0: stopwatch = 0
                #if state == 0 and go > theta and stopwatch == 0:
                    #     stopwatch = t
                #     print("table set stopwatch")
                if state == 0 and go > theta and stopwatch == 0:
                    words_list = list(words.items())
                    key, _ = words_list[np.argmax([func @ w.v for _, w in words_list])]
                    function = bindings[key]
                    to_controller = function.v
                    ctrl_sig = 1
                    stopwatch = t
                    state = 1
                    #print("table retreived func, pulsing control signal")
                elif state == 1 and go > theta and t > stopwatch + t_ctrl:
                    ctrl_sig = 0
                    state = 2
                    #print("table done pulsing")
                elif state == 2 and go > theta and t > stopwatch + t_busy:
                    state = 3
                    #print("table no longer busy")
                elif state == 3 and go < theta and t > stopwatch + t_done:
                    to_controller = np.zeros(d)
                    state = 0
                    stopwatch = 0
                    #print("table reset")
                #print(f'{t:.2f} {state} {go:.2f} {stopwatch + t_busy:.2f} {ctrl_sig}')
                return np.concatenate([to_controller, [ctrl_sig, state == 3]])
            return prog_table
        with self:
            self.func_key = nengo.Node(size_in=d) 
            program_table = nengo.Node(size_in=d+1, 
                                       output=make_prog_table(self.words, self.bindings), 
                                       label="program table"
                                       )
            nengo.Connection(self.func_key, program_table[:d])
            nengo.Connection(self.input, program_table[-1])
        
            self.retrieved_func = nengo.Node(size_in=d)
            self.ctrl_sigout = nengo.Node(size_in=1)
            self.holo_node = nengo.Node(size_out=d, output=lambda _: self.holo)
            nengo.Connection(program_table[:d], self.retrieved_func)
            nengo.Connection(program_table[-2], self.ctrl_sigout)
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
        !           F_PUT
        @           F_PEEP
        +           F_ADD
        -           F_SUB
        0=          F_ISZERO
        drop        F_DROP
        dup         F_DUP
        swap        F_SWAP
        if          F_IF
        else        F_ELSE
        then        F_THEN
    """
    
    voc_items = ["R_LEFT", "R_RIGHT", "R_PHI", "T_NIL",
                 "S_PUSH", "S_POP", "S_PEEK", "S_DUMP", 
                 "S_CODE_ERR_STACKEMPTY", 'S_CODE_HALT', 
                 "S_ORIG_2", "S_ORIG_1",
                 "TRUE", "FALSE"
                 ] 
    voc.populate("; ".join(voc_items))

    # Need to add ROT command too, but will talk later about that, if we are doing return stack operations.
    
    #registers = RegisterBank(
    #        ['R1', 'R2', 'R3', 'R4', 'R5', 
    #         'I1', 'I2', 'I3', 
    #         'O1', 'O2', 'O3']
    #        )

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
    data_stack = SimpleStack(label="data_stack")
    ctrl_flow_stack = SimpleStack(label="ctrl_flow_stack")
        
    wds_circuits = spa.Network("Words Circuits")
    with wds_circuits:
        wds_circuits.circuits_dict = {
            "F_DUP":     DupCircuit(data_stack.stack, vocab=voc, label="DUP Circuit"),
            "F_DROP":    DropCircuit(data_stack.stack, vocab=voc, label="DROP Circuit"), 
            "F_SWAP":    SwapCircuit(data_stack.stack, vocab=voc, label="SWAP Circuit"),
            #"F_ADD":     AddCircuit(data_stack.stack, vocab=voc, label="ADD Circuit"),
            #"F_SUB":     SubCircuit(data_stack.stack, vocab=voc, label="SUB Circuit"),
            #"F_ISZERO":  IsZeroCircuit(data_stack.stack, vocab=voc, label="0= Circuit"),
            #"F_PEEP":    PeepCircuit(data_stack.stack, registers, vocab=voc, label="@ Circuit"),
            #"F_PUT":     PutCircuit(data_stack.stack, registers, vocab=voc, label="! Circuit"),
            #"F_FUNC":    FuncCircuit([], [], vocab=voc, label=": Circuit"),  # Need compilation state
            #"F_END":     EndCircuit([], [], vocab=voc, label="; Circuit"),   # Need compilation state  
            "F_IF":      IfCircuit(data_stack.stack, ctrl_flow_stack.stack, vocab=voc, label="IF Circuit"),
            "F_ELSE":       ElseCircuit(ctrl_flow_stack.stack, vocab=voc, label="ELSE Circuit"),
            "F_THEN":    ThenCircuit(ctrl_flow_stack.stack, vocab=voc, label="THEN Circuit"),
        }

        voc.populate('; '.join(list(wds_circuits.circuits_dict.keys())))
        
        fruits = ['APPLE', 'BANANA', 'CHERRY', 'DURIAN', 'ELDERBERRY', 'FIG', 'GUAVA', 'HYUGANATSU', 
                  'IMBE', 'JACKFRUIT', 'KUMQUAT', 'LYCHEE', 'MANGO', 'NECTARINE', 'ORANGE', 'PINEAPPLE', 
                  'QUINCE', 'RASPBERRY', 'SASKATOON', 'TANGERINE', 'UNNAB', 'VOAVANGA', 'WATERMELON', 
                  'XOCONOSTLE', 'YUZU', 'ZWETSCHGE']
        voc.populate('; '.join(fruits))

        table = {'FRUITSWAP': ['LYCHEE', 'MANGO', 'F_SWAP'],
                 'FRUITREADY': ['TRUE', 'F_IF', 'FRUITSWAP', 'F_ELSE', 'LYCHEE', 'F_THEN', 'BANANA']
                 }                


        wds_circuits.user_func_circuit = UserFuncCircuit(user_table=table)

        print(wds_circuits.user_func_circuit.table)
        print([v.name for v in wds_circuits.user_func_circuit.bindings.values()])

    call_stack = SimpleStack(label="call_stack")
    return_stack = SimpleStack(label="return_stack")

    assert ctrl_flow_stack.stack is not call_stack.stack

    # holo = sum([voc[c].v for c in circuits_dict.keys()])
    # print(np.sqrt(len(circuits_dict.values())), 
    #       np.linalg.norm(holo),
    #      [voc[c].v @ holo for c in circuits_dict.keys()])
    
    lis = cons(voc['CHERRY'], cons(voc["APPLE"], voc["BANANA"]))
    listail1 = cons(voc["APPLE"], voc["BANANA"])
    listail2 = cons(voc["BANANA"],voc["T_NIL"])
    
    two = cons(voc['T_NIL'], cons(voc['T_NIL'], voc['T_NIL']))
    three = cons(voc['T_NIL'], two)  # Using existing list
    result = add_list_numbers(two, three, voc)
    #print(count_list_depth(result, voc))  # Should print 5
    
    test_program = make_list(["CHERRY", "FRUITSWAP", "CHERRY", "MANGO"], vocab=voc)

    
    #voc.add(test_program.name, test_program.v)
    
    inp = spa.State(voc)
    out = spa.State(voc)
    c_inp = spa.State(voc)
    
    
    nengo.Connection(inp.output, data_stack.input)
    nengo.Connection(data_stack.output, out.input)
    nengo.Connection(c_inp.output, ctrl_flow_stack.input)

    control_unit = ControlUnit(d=d, theta=theta, items=test_program, label="ControlUnit Network", vocab=voc, assoc_memory=assoc, circuits=wds_circuits.circuits_dict)

    
    nengo.Connection(control_unit.to_data_stack, inp.input)
    
    nengo.Connection(control_unit.clock_trigger, data_stack.sigin)
    nengo.Connection(control_unit.clock_trigger, ctrl_flow_stack.sigin)

    nengo.Connection(control_unit.to_call_stack, call_stack.input)
    nengo.Connection(control_unit.call_stack_sigout, call_stack.sigin)
    
    
    busy_node = nengo.Node(output=make_busy_signal(), size_in=2, size_out=1, label="busy_node")
    nengo.Connection(busy_node, control_unit.word_busy)

    dispatcher = Dispatcher(wds_circuits, busy_node, vocab=voc)

    nengo.Connection(control_unit.to_dispatcher, dispatcher.input)
    nengo.Connection(control_unit.to_dispatcher, wds_circuits.user_func_circuit.func_key)
    nengo.Connection(wds_circuits.user_func_circuit.holo_node, dispatcher.holo_node)

    nengo.Connection(wds_circuits.user_func_circuit.holo_node, control_unit.holo_node)
    nengo.Connection(wds_circuits.user_func_circuit.ctrl_sigout, control_unit.func_ctrl_sigin)
    nengo.Connection(wds_circuits.user_func_circuit.retrieved_func, control_unit.from_user_func)
    nengo.Connection(call_stack.sigout, control_unit.call_stack_sigin)
    nengo.Connection(call_stack.output, control_unit.from_call_stack)
    nengo.Connection(control_unit.func_ctrl_done, wds_circuits.user_func_circuit.output)
    nengo.Connection(control_unit.flag_control[d:d*2], c_inp.input)
    nengo.Connection(ctrl_flow_stack.output, control_unit.flag_control[d:d*2])

    nengo.Connection(control_unit.to_return_stack, return_stack.input)
    nengo.Connection(control_unit.call_stack_sigout, return_stack.sigin)
#    function_decoder = spa.State(voc)
#    nengo.Connection(wds_circuits.user_func_circuit.retrieved_func, function_decoder.input)

    test_call_stack_out = spa.State(voc)
    nengo.Connection(call_stack.output, test_call_stack_out.input)

    test_call_stack_in = spa.State(voc)
    nengo.Connection(call_stack.input, test_call_stack_in.input)

    test_func_ctrl = spa.State(voc)
    nengo.Connection(control_unit.func_ctrl[:d], test_func_ctrl.input)
