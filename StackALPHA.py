import nengo
import nengo_spa as spa
from numpy.linalg import norm
import numpy as np

class HighSpeedOscillator:
    def __init__(self, network, speed=1.0, label="Oscillator"):
        self.speed = speed
        self.label = label
        
        with network:
            self.subnet = nengo.Network(label=label)
            
        with self.subnet:
            self.ensemble = nengo.Ensemble(
                n_neurons=1000,
                dimensions=3,
                radius=1.5,
                label="oscillator_core"
            )
            
            # Fixed parameters
            self.synapse = 0.005
            
            def dynamics(x):
                r, s = 1.0, 50 * x[2]
                return [
                    self.synapse * -x[1] * s + x[0] * (r - x[0]**2 - x[1]**2) + x[0],
                    self.synapse * x[0] * s + x[1] * (r - x[0]**2 - x[1]**2) + x[1],
                ]
            nengo.Connection(
                self.ensemble,
                self.ensemble[:2],
                synapse=self.synapse,
                function=dynamics
            )
            
            self.speed_node = nengo.Node(
                output=self.speed,
                label="speed_input"
            )
            speed_ens = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="speed_encoder"
            )
            nengo.Connection(self.speed_node, speed_ens)
            nengo.Connection(speed_ens, self.ensemble[2])
            
            self.output = nengo.Node(size_in=2, label="continuous_output")
            self.ticks = nengo.Node(size_in=1, label="pulse_output")
            
            nengo.Connection(self.ensemble[:2], self.output)
            
            tick_ens = nengo.Ensemble(100, 1, label="tick_detector")
            nengo.Connection(
                self.ensemble[0],
                tick_ens,
                function=lambda x: 3 if x > 0.8 else -1.0
            )
            nengo.Connection(tick_ens, self.ticks)
            
    def connect_to(self, target, mode="continuous", transform=None):
        with self.subnet:
            if mode == "continuous":
                nengo.Connection(self.output, target, transform=transform)
            elif mode == "pulse":
                nengo.Connection(self.ticks, target, transform=transform)

class ANDGate(nengo.Network):
    def __init__(self, n_neurons=200, dimensions=1, label=None):
        super().__init__(label=label)
        
        with self:
            self.input_a = nengo.Node(size_in=dimensions)
            self.input_b = nengo.Node(size_in=dimensions)
            
            self.thresh_node_a = nengo.Node(lambda t, x: 1 if x > 0 else -1, size_in=dimensions)
            self.thresh_node_b = nengo.Node(lambda t, x: 1 if x > 0 else -1, size_in=dimensions)
            
            nengo.Connection(self.input_a, self.thresh_node_a)
            nengo.Connection(self.input_b, self.thresh_node_b)
            
            self.thresh_a = nengo.Ensemble(n_neurons, dimensions)
            self.thresh_b = nengo.Ensemble(n_neurons, dimensions)
            
            nengo.Connection(self.thresh_node_a, self.thresh_a)
            nengo.Connection(self.thresh_node_b, self.thresh_b)
            
            self.combiner = nengo.Ensemble(n_neurons, 2)
            nengo.Connection(self.thresh_a, self.combiner[0])
            nengo.Connection(self.thresh_b, self.combiner[1])
            
            self.output = nengo.Ensemble(n_neurons*2, dimensions)
            nengo.Connection(self.combiner, self.output,
                            function=lambda x: 1 if np.min(x) > 0 else -1)
            
            self.output_node = nengo.Node(size_in=dimensions)
            nengo.Connection(self.output, self.output_node,
                            function=lambda x: 1 if x > 0 else -1)
                            

class XORGate(nengo.Network):
    def __init__(self, n_neurons=200, dimensions=1, label=None):
        super().__init__(label=label)
        
        with self:
            self.input_a = nengo.Node(size_in=dimensions)
            self.input_b = nengo.Node(size_in=dimensions)
            
            self.thresh_a = nengo.Ensemble(n_neurons, dimensions)
            self.thresh_b = nengo.Ensemble(n_neurons, dimensions)
            
            self.thresh_node_a = nengo.Node(lambda t, x: 1 if x > 0 else -1, size_in=dimensions)
            self.thresh_node_b = nengo.Node(lambda t, x: 1 if x > 0 else -1, size_in=dimensions)            
            
            nengo.Connection(self.input_a, self.thresh_node_a)
            nengo.Connection(self.input_b, self.thresh_node_b)
            
            nengo.Connection(self.thresh_node_a, self.thresh_a)
            nengo.Connection(self.thresh_node_b, self.thresh_b)
            
            self.combiner = nengo.Ensemble(n_neurons, 2)
            nengo.Connection(self.thresh_a, self.combiner[0])
            nengo.Connection(self.thresh_b, self.combiner[1])
            
            self.output = nengo.Ensemble(n_neurons*2, dimensions)
            nengo.Connection(self.combiner, self.output,
                            function=lambda x: 1 if np.prod(x) < 0 else -1)
            
            self.output_node = nengo.Node(size_in=dimensions)
            nengo.Connection(self.output, self.output_node,
                            function=lambda x: 1 if x > 0 else -1)

class Incrementer(nengo.Network):
    def __init__(self, n_bits, label=None):
        super().__init__(label=label)
        
        with self:
            self.input = nengo.Node(size_in=n_bits, label="Input Bits")
            self.output = nengo.Node(size_in=n_bits+1, label="Output (+Overflow)")
            
            self.carry_chain = []
            prev_carry = nengo.Node(output=1, label="Initial Carry") 
            
            for i in range(n_bits):
                bit = nengo.Node(size_in=1, label=f"Bit {i} Input")
                nengo.Connection(self.input[i], bit)
                
                xor = XORGate(label=f"Bit{i}_XOR")
                and_gate = ANDGate(label=f"Bit{i}_AND")
                
                nengo.Connection(bit, xor.input_a)
                nengo.Connection(prev_carry, xor.input_b)
                nengo.Connection(bit, and_gate.input_a)
                nengo.Connection(prev_carry, and_gate.input_b)
                
                prev_carry = and_gate.output_node
                self.carry_chain.append(prev_carry)
                
                nengo.Connection(xor.output_node, self.output[i])
            
            nengo.Connection(prev_carry, self.output[n_bits])

    def connect_input(self, source):
        nengo.Connection(source, self.input)
    
    def connect_output(self, target):
        nengo.Connection(self.output, target)
        
class Decrementer(nengo.Network):
    def __init__(self, n_bits, label=None):
        super().__init__(label=label)
        
        with self:
            self.input = nengo.Node(size_in=n_bits, label="Input Bits")
            self.output = nengo.Node(size_in=n_bits+1, label="Output (+Underflow)")
            
            self.borrow_chain = []
            prev_borrow = nengo.Node(output=1, label="Initial Borrow")
            
            for i in range(n_bits):
                bit = nengo.Node(size_in=1, label=f"Bit {i} Input")
                nengo.Connection(self.input[i], bit)
                
                xor = XORGate(label=f"Bit{i}_XOR")
                and_gate = ANDGate(label=f"Bit{i}_AND")
                
                inverted_bit = nengo.Node(lambda t, x: -x, size_in=1, label=f"Bit {i} Inverted")
                nengo.Connection(bit, inverted_bit)
                
                nengo.Connection(bit, xor.input_a)
                nengo.Connection(prev_borrow, xor.input_b)
                nengo.Connection(inverted_bit, and_gate.input_a)
                nengo.Connection(prev_borrow, and_gate.input_b)
                
                prev_borrow = and_gate.output_node
                self.borrow_chain.append(prev_borrow)
                
                nengo.Connection(xor.output_node, self.output[i])
            
            nengo.Connection(prev_borrow, self.output[n_bits])

    def connect_input(self, source):
        nengo.Connection(source, self.input)
    
    def connect_output(self, target):
        nengo.Connection(self.output, target)

class Demux(spa.Network):
    """A single demultiplexer node."""
    def __init__(self, signals, input_vocab):
        super().__init__()
        
        with self:
            self.control_input = spa.State(signals, subdimensions=1)
            
            self.input = spa.State(input_vocab)
            self.top = spa.State(input_vocab)
            self.bot = spa.State(input_vocab)
        
            self.switch = spa.ActionSelection()
            with self.switch:
                spa.ifmax(
                    "up", 
                    spa.dot(self.control_input, signals["ZERO"]),
                    self.input >> self.top
                )
                spa.ifmax(
                    "down", 
                    spa.dot(self.control_input, signals["ONE"]),
                    self.input >> self.bot
                )

class DemuxTree(spa.Network):
    """A hierarchical tree of demultiplexers."""
    def __init__(self, n, signals, input_vocab):
        try:
            assert(n % 2 == 0)
        except:
            raise TypeError("n must be divisible by 2")
            
        super().__init__()
        
        with self:
            self.output_array = spa.Network()
            with self.output_array as oa:
                oa.cells = [spa.State(input_vocab) for _ in range(n)]
        
            self.layers = [self.output_array]
        
            levels = int(np.log2(n))
            for level in range(levels):
                new_layer = spa.Network()
                with new_layer as l:
                    l.cells = [Demux(signals, input_vocab) for _ in range(n // (2 ** (level + 1)))]
                
                for i, cell in enumerate(new_layer.cells):
                    cell.top >> self.layers[-1].cells[i * 2].input
                    cell.bot >> self.layers[-1].cells[i * 2 + 1].input
            
                self.layers.append(new_layer)
                
            self.layers = self.layers[::-1]
            self.input = self.layers[0].cells[0].input

            self.control_inputs = [spa.State(signals, subdimensions=1) for _ in range(levels)]
            for level, control_input in enumerate(self.control_inputs):
                for cell in self.layers[level].cells:
                    nengo.Connection(control_input.output, cell.control_input.input)

class Mux(spa.Network):
    """A single multiplexer node."""
    def __init__(self, signals, output_vocab):
        super().__init__()
        
        with self:
            self.control_input = spa.State(signals, subdimensions=1)
            
            self.top = spa.State(output_vocab)
            self.bot = spa.State(output_vocab)
            self.output = spa.State(output_vocab)
        
            self.switch = spa.ActionSelection()
            with self.switch:
                spa.ifmax(
                    "up", 
                    spa.dot(self.control_input, signals["ZERO"]),
                    self.top >> self.output
                )
                spa.ifmax(
                    "down", 
                    spa.dot(self.control_input, signals["ONE"]),
                    self.bot >> self.output
                )

class MuxTree(spa.Network):
    """A hierarchical tree of multiplexers."""
    def __init__(self, n, signals, output_vocab):
        try:
            assert(n % 2 == 0)
        except:
            raise TypeError("n must be divisible by 2")
            
        super().__init__()
        
        with self:
            self.input_array = spa.Network()
            with self.input_array as ia:
                ia.cells = [spa.State(output_vocab) for _ in range(n)]
        
            self.layers = [self.input_array]
        
            levels = int(np.log2(n))
            for level in range(levels):
                new_layer = spa.Network()
                with new_layer as l:
                    l.cells = [Mux(signals, output_vocab) for _ in range(n // (2 ** (level + 1)))]
                
                for i, cell in enumerate(new_layer.cells):
                    self.layers[-1].cells[i * 2].output >> cell.top
                    self.layers[-1].cells[i * 2 + 1].output >> cell.bot
            
                self.layers.append(new_layer)
                
            self.layers = self.layers[::-1]
            self.output = self.layers[0].cells[0].output

            self.control_inputs = [spa.State(signals, subdimensions=1) for _ in range(levels)]
            for level, control_input in enumerate(self.control_inputs):
                for cell in self.layers[level].cells:
                    nengo.Connection(control_input.output, cell.control_input.input)

class MemoryCell(spa.Network):
    def __init__(self, vocab, label=None):
        super().__init__(label=label)
        
        with self:
            self.combined_input = spa.State(vocab, label=f"{label} Combined Input")
            self.memory = spa.State(vocab, feedback=1, label=f"{label} Memory")
            self.output = spa.State(vocab, label=f"{label} Output")
            
            self.action = spa.ActionSelection()
            with self.action:
                spa.ifmax(
                    "READ",
                    spa.dot(self.combined_input, vocab["READ"]),
                    self.memory >> self.output
                )
                spa.ifmax(
                    "WRITE",
                    spa.dot(self.combined_input, vocab["WRITE"]),
                    (self.combined_input - vocab["WRITE"] - self.memory) >> self.memory
                )

            dim = vocab.dimensions
            def vcos(t, x):
                mag = norm(x[:dim]) * norm(x[dim:])
                if mag == 0: return 0
                
                return (x[:dim].dot(x[dim:]))/mag
            stable = nengo.Node(size_out=1, size_in=2*dim, output=vcos)
            nengo.Connection(self.memory.output, stable[:dim])
            nengo.Connection(self.memory.output, stable[dim:],synapse=0.4)
            
            threshold = 0.80
            threshold_node = nengo.Node(
                size_in=1,
                size_out=1,
                output=lambda t, x: 1 if x < threshold else -1
            )
            nengo.Connection(stable, threshold_node)
            
            self.threshold_output = threshold_node


class AddressController(nengo.Network):
    def __init__(self, address_bits, label=None):
        super().__init__(label=label)
        
        self.address_bits = address_bits
        
        with self:
            def address_node_func(t, x):
                if not hasattr(address_node_func, 'stored_value'):
                    address_node_func.stored_value = np.array([1]*address_bits)
                if x.size > 0:
                    address_node_func.stored_value = x
                return address_node_func.stored_value
            
            self.current_address = nengo.Node(address_node_func,
                                            size_in=address_bits,
                                            size_out=address_bits,
                                            label="Current Address")
            
            self.do_increment = nengo.Node(size_in=1, size_out=1, label="Increment Signal")
            self.do_decrement = nengo.Node(size_in=1, size_out=1, label="Decrement Signal")
            
            def init_zero(t):
                return 0
            nengo.Connection(nengo.Node(init_zero), self.do_increment)
            nengo.Connection(nengo.Node(init_zero), self.do_decrement)
            
            self.incrementer = Incrementer(address_bits, label="Incrementer")
            self.decrementer = Decrementer(address_bits, label="Decrementer")
            
            def update_address(t, x):
                current = x[:address_bits]
                inc_signal = x[address_bits]
                dec_signal = x[address_bits+1]
                inc_result = x[address_bits+2:2*address_bits+2]
                dec_result = x[2*address_bits+2:3*address_bits+2]
                if t <= 0.01:  
                    return np.array([1.0]*address_bits)                
                if inc_signal > 0:
                    return inc_result
                elif dec_signal > 0:
                    return dec_result
                return current
            
            self.address_updater = nengo.Node(update_address,
                                             size_in=3*address_bits+2,
                                             size_out=address_bits,
                                             label="Address Updater")
            
            nengo.Connection(self.current_address, self.incrementer.input)
            nengo.Connection(self.current_address, self.decrementer.input)
            
            nengo.Connection(self.current_address, self.address_updater[:address_bits])
            nengo.Connection(self.do_increment, self.address_updater[address_bits])
            nengo.Connection(self.do_decrement, self.address_updater[address_bits+1])
            nengo.Connection(self.incrementer.output[:address_bits], 
                           self.address_updater[address_bits+2:2*address_bits+2])
            nengo.Connection(self.decrementer.output[:address_bits],
                           self.address_updater[2*address_bits+2:3*address_bits+2])
            nengo.Connection(self.address_updater, self.current_address, synapse=0.01)
            
            self.control = nengo.Node(size_in=2, label="Control Input")
            nengo.Connection(self.control[0], self.do_increment)
            nengo.Connection(self.control[1], self.do_decrement)
            
            self.output = nengo.Node(size_in=address_bits, size_out=address_bits)
            nengo.Connection(self.current_address, self.output)
            
            def address_change_func(t, x):
                current = x[:address_bits]
                delayed = x[address_bits:]
                mag = np.linalg.norm(current) * np.linalg.norm(delayed)
                if mag == 0: 
                    return 0
                similarity = np.dot(current, delayed) / mag
                return 1.0 if similarity < 0.999 else -1.0
                
            self.address_change = nengo.Node(address_change_func,
                                           size_in=address_bits*2,
                                           size_out=1,
                                           label="Address Change Detector")
            
            nengo.Connection(self.current_address, self.address_change[:address_bits])
            nengo.Connection(self.current_address, self.address_change[address_bits:], 
                           synapse=0.05)
            

class HierarchicalMemoryController(spa.Network):
    """Controls routing of data between input/output and memory cells using hierarchical trees."""
    def __init__(self, num_cells, vocab, signals):
        super().__init__(label="Hierarchical Memory Controller")
        self.num_cells = num_cells
        self.n = int(np.ceil(np.log2(num_cells)))

        with self:
            self.combined_input = spa.State(vocab, label="Combined Input/Control Signal")
            self.output_cell = spa.State(vocab, label="Output Cell")

            self.memory_cells = [MemoryCell(vocab, label=f"Memory Cell {i}") for i in range(num_cells)]
            self.demux_tree = DemuxTree(num_cells, signals, vocab)
            self.mux_tree = MuxTree(num_cells, signals, vocab)

            for i, cell in enumerate(self.memory_cells):
                nengo.Connection(self.demux_tree.output_array.cells[i].output, cell.combined_input.input)
            for i, cell in enumerate(self.memory_cells):
                nengo.Connection(cell.output.output, self.mux_tree.input_array.cells[i].input)
            nengo.Connection(self.mux_tree.output.output, self.output_cell.input)
            nengo.Connection(self.combined_input.output, self.demux_tree.input.input)

            self.control_inputs = [spa.State(signals, subdimensions=1) for _ in range(self.n)]
            for level, control_input in enumerate(self.control_inputs):
                nengo.Connection(control_input.output, self.demux_tree.control_inputs[level].input)
                nengo.Connection(control_input.output, self.mux_tree.control_inputs[level].input)

            self.busy_write_signal = nengo.Node(
                size_in=num_cells,
                size_out=1,
                output=lambda t, x: 1.0 if np.any(x > 0.5) else -1.0,
                label="busy_write_signal"
            )
            for i in range(num_cells):
                nengo.Connection(
                    self.memory_cells[i].threshold_output,
                    self.busy_write_signal[i],
                    synapse=None
                )

            dim = vocab.dimensions
            self.stable = nengo.Node(size_out=1, size_in=dim, output=lambda t, x: np.dot(x, x))
            nengo.Connection(self.output_cell.output, self.stable)
            
            # Add threshold node
            threshold = 0.62
            self.busy_read_signal = nengo.Node(
                size_in=1,
                size_out=1,
                output=lambda t, x: 1 if x[0] < threshold else 0,
                label="busy_read_signal"
            )
            nengo.Connection(self.stable, self.busy_read_signal)

    def connect_address_controller(self, address_controller):
        for i in range(self.n):
            def make_control_func():
                return lambda t, x: 1 if x > 0 else -1
            control_node = nengo.Node(make_control_func(), size_in=1, label=f"AddressBit{i}_ControlConverter")
            nengo.Connection(address_controller.output[self.n - 1 - i], control_node)
            nengo.Connection(control_node, self.control_inputs[i].input)

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
            nengo.Connection(nengo.Node(output=bias), self.compare)
            
            self.output = self.compare

class StackController(spa.Network):
    def __init__(self, memory_controller, address_controller, vocab, signals):
        super().__init__(label="Stack Controller")
        
        with self:
            memory_controller.connect_address_controller(address_controller)
            
            self.push_edge = RisingEdgeDetector(label="Push Edge Detector")
            self.pop_edge = RisingEdgeDetector(label="Pop Edge Detector")

            self.push_cmd = nengo.Node(size_in=1, label="Push Command")
            self.pop_cmd = nengo.Node(size_in=1, label="Pop Command")
            self.data_input = spa.State(vocab, label="Data Input")

            nengo.Connection(self.push_cmd, self.push_edge.input)
            nengo.Connection(self.pop_cmd, self.pop_edge.input)
            
            self.busywrite_fall_edge = FallingEdgeDetector(label="Busy Write Falling Edge Detector",tau=0.02, bias=-1)
            nengo.Connection(memory_controller.busy_write_signal, self.busywrite_fall_edge.input)
            
            self.busyread_fall_edge = FallingEdgeDetector(label="Busy Read Falling Edge Detector",tau=0.04, bias=-0.5)
            nengo.Connection(memory_controller.busy_read_signal, self.busyread_fall_edge.input)

            def falling_edge_latch(t, x):
                if t <= .1:
                    return -1
                sig_s = x[0]
                sig_r = x[1]
                state = x[2]
                
                if sig_s > 0:
                    return 1
                elif sig_r > 0:
                    return -1
                else:
                    return 1 if state > 0 else -1
            
            self.clean_increment = nengo.Node(output=falling_edge_latch,size_in=3, size_out=1)
            self.clean_push = nengo.Node(output=falling_edge_latch,size_in=3, size_out=1)
            self.clean_pop = nengo.Node(output=falling_edge_latch,size_in=3, size_out=1)
            self.exclude_node = nengo.Node(size_in=2, size_out=1, output = lambda t, x: np.prod(x))
            
            nengo.Connection(self.clean_push, self.exclude_node[0])
            nengo.Connection(self.clean_pop, self.exclude_node[1])
            
            nengo.Connection(self.push_edge.output, self.clean_increment[0])
            nengo.Connection(self.push_edge.output, address_controller.do_increment)
            nengo.Connection(self.busyread_fall_edge.output, address_controller.do_decrement)
            nengo.Connection(self.pop_edge.output, self.clean_pop[0])
            
            nengo.Connection(self.busywrite_fall_edge.output, self.clean_push[1])
            nengo.Connection(address_controller.address_change, self.clean_increment[1])
            nengo.Connection(self.busyread_fall_edge.output, self.clean_pop[1])
            
            nengo.Connection(self.clean_increment, self.clean_increment[2])
            nengo.Connection(self.clean_push, self.clean_push[2])
            nengo.Connection(self.clean_pop, self.clean_pop[2])
            
            self.busyincrement_fall_edge = FallingEdgeDetector(label="Busy Increment Falling Edge Detector",tau=0.02, bias=-1)
            
            nengo.Connection(self.clean_increment, self.busyincrement_fall_edge.input)
            nengo.Connection(self.busyincrement_fall_edge.output, self.clean_push[0])
            
            self.input_selector = spa.ActionSelection()
            with self.input_selector:
                spa.ifmax(
                    "ReadWrite",
                    self.exclude_node,
                )
                spa.ifmax(
                    "Write",
                    self.clean_push,
                    (self.data_input + vocab["WRITE"]) >> memory_controller.combined_input
                )
                spa.ifmax(
                    "Read",
                    self.clean_pop,
                    vocab["READ"] >> memory_controller.combined_input
                )
            

    def connect_write_trigger(self, source):
        """Connect external write trigger signal"""
        nengo.Connection(source, self.push_cmd)
        
    def connect_read_trigger(self, source):
        """Connect external read trigger signal"""
        nengo.Connection(source, self.pop_cmd)
        
    def connect_data_input(self, source):
        """Connect external data source for push operations"""
        nengo.Connection(source.output, self.data_input.input)
        

model = spa.Network()
with model:
    vocab = spa.Vocabulary(256, strict=False)
    vocab.populate("READ; WRITE; ONE; ZERO")
    
    signals = spa.Vocabulary(1, strict=False)
    signals.add("ONE", [1])
    signals.add("ZERO", [-1])
    
    num_cells = 4
    memory = HierarchicalMemoryController(num_cells=num_cells, vocab=vocab, signals=signals)
    
    address_controller = AddressController(address_bits=int(np.ceil(np.log2(num_cells))),
                                          label="Address Controller")
    

    stack = StackController(memory, address_controller, vocab, signals)
    
    write_trigger = nengo.Node(lambda t: 0 if t > 0.5 else 0)
    read_trigger = nengo.Node(lambda t: 0 if t > 1.0 else 0)
    data_source = spa.State(vocab)

    stack.connect_write_trigger(write_trigger)
    stack.connect_read_trigger(read_trigger)
    stack.connect_data_input(data_source)
