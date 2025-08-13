import numpy as np
import nengo
import nengo_spa as spa

theta = 0.3
t_delay = 0.1
t_pulse = 0.2

# if sub_req is false, sigout should pulse on each put and we should pop at regular intervals
# if pub_req is false, we should put at regular intervals and pulse sigout whenever something is popped
class RingBuffer(spa.Network):
    def __init__(self, buf_size, dim, pub=None, sub=None, pub_req=True, 
                 sub_req=True, interval=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.buf_size = buf_size
        self._width = int(np.log10(self.buf_size-1)) + 1
        self._buffer = np.zeros((self.buf_size, dim))
        self._read_head = self.buf_size - 1
        self._write_head = 0
        self._iter_flag = False
        
        assert self.label is not None and self.label.isalnum() and self.label[0].isalpha()

        self.pub = pub
        self.sub = sub

        if not self.pub:
            with self:
                self.pub = spa.Network("Publisher")
                self._generate_pub()
        else:
            self._generate_pub()
        
        if not self.sub:
            with self:
                self.sub = spa.Network("Subscriber")
                self._generate_sub()
        else:
            self._generate_sub()

        with self:
            def make_write_state_machine(pub_req):
                state = 0
                stopwatch = 0
                def write_state_machine(t, x):
                    nonlocal state, stopwatch
                    from_pub = x[:self.dim]
                    pub_sigin = x[-1]

                    if state == 0 and pub_sigin > theta:
                        state = 1
                        self.put(from_pub)
                        stopwatch = t
                        print(self, 'put!')
                    elif state == 1 and t > stopwatch + t_delay:
                        state = 2
                    elif state == 2 and t > stopwatch + t_delay + t_pulse:
                        state = 0
                        stopwatch = 0

                    return [0, state==2]

                return write_state_machine

            def make_read_state_machine(sub_req):
                state = 0
                stopwatch = 0
                to_return = np.zeros(self.dim)
                def read_state_machine(t, x):
                    nonlocal state, stopwatch, to_return

                    sub_sigin = x[-1]

                    if state == 0 and sub_sigin > theta:
                        try:
                            to_return[:] = self.pop()
                            state = 1
                        except IndexError:
                            to_return[:] = 0
                            state = -1
                        stopwatch = t
                        print(self, "popped!")
                    elif state == 1 and t > stopwatch + t_delay:
                        state = 2
                    elif state == -1 and t > stopwatch + t_delay:
                        state = -2
                    elif np.abs(state) == 2 and t > stopwatch + t_delay + t_pulse:
                        state = 0
                        stopwatch = 0
                        to_return[:] = 0

                    return np.concatenate([to_return, [0, state//2]])

                return read_state_machine



            read_controller = nengo.Node(size_in=1,
                                         size_out=self.dim+2,
                                         output=make_read_state_machine(sub_req),
                                         label="read_controller")

            write_controller = nengo.Node(size_in=self.dim+1,
                                         size_out=2,
                                         output=make_write_state_machine(pub_req),
                                         label="write_controller")

            nengo.Connection(self.pub.publisher.output, write_controller[:self.dim])
            nengo.Connection(read_controller[:self.dim], self.sub.subscriber.input)

            nengo.Connection(self.pub.publisher.sigin, write_controller[-1])
            nengo.Connection(self.sub.subscriber.sigin, read_controller)
            nengo.Connection(write_controller[-1], self.pub.publisher.sigout)
            nengo.Connection(write_controller[-2], self.sub.subscriber.sigout)
            nengo.Connection(read_controller[-1], self.sub.subscriber.sigout)
            nengo.Connection(read_controller[-2], self.pub.publisher.sigout)
       
    def _generate_pub(self):
        with self.pub:
            publabel = self.label + "_publisher"
            self.pub.publisher = spa.Network(label=publabel)
            publisher = self.pub.publisher
            setattr(self.pub, publabel, publisher)

            with publisher:
                publisher.register = spa.State(self.dim)
                publisher.input = nengo.Node(size_in=self.dim)
                publisher.output = nengo.Node(size_in=self.dim)
                publisher.sigin = nengo.Node(size_in=1)
                publisher.sigout = nengo.Node(size_in=1)

                nengo.Connection(publisher.input, publisher.register.input)
                nengo.Connection(publisher.register.output, publisher.output)

    def _generate_sub(self):
        with self.sub:
            sublabel = self.label + "_subscriber"
            self.sub.subscriber = spa.Network(label=sublabel)
            subscriber = self.sub.subscriber
            setattr(self.sub, sublabel, subscriber)

            with subscriber:
                subscriber.register = spa.State(self.dim)
                subscriber.input = nengo.Node(size_in=self.dim)
                subscriber.output = nengo.Node(size_in=self.dim)
                subscriber.sigin = nengo.Node(size_in=1)
                subscriber.sigout = nengo.Node(size_in=1)

                nengo.Connection(subscriber.input, subscriber.register.input)
                nengo.Connection(subscriber.register.output, subscriber.output)


    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_flag:
            raise StopIteration
        return self.pop()

    def put(self, value):
        self._buffer[self._write_head, :] = value[:]
        self._write_head = (self._write_head + 1) % self.buf_size
        if self._write_head == (self._read_head + 1) % self.buf_size:
            self._read_head = (self._read_head + 1) % self.buf_size
        self._iter_flag = False

    def pop(self):
        if self._read_head != (self._write_head - 1) % self.buf_size:
            self._read_head = (self._read_head + 1) % self.buf_size
        else:
            self._iter_flag = True
        
        if self._iter_flag:
            raise IndexError("Queue is empty!")
        
        item = self._buffer[self._read_head]

        return item

    def _format_array(self, arr):
        return "  ".join([f'{item:5.2f}' for item in arr])

    def __str__(self):
        rep = ""
        left = ""
        right = ""
        padding = "             " + " "*self._width
        if self.dim > 8:
            if self.buf_size > 8:
                for i, row in enumerate(self._buffer[:3]):
                    if i == self._write_head:
                        left = f"[{i:{self._width}d}] writer >>>"
                    else:
                        left = padding
                    if i == self._read_head:
                        right = f"<<< reader [{i:{self._width}d}]" 
                    else:
                        right = ""
                    rep += ' {} [{}  ...  {}] {}\n'.format(left,
                                                        self._format_array(row[:3]), 
                                                        self._format_array(row[-3:]),
                                                        right)


                if ((self._read_head > 2 and self._read_head < self.buf_size - 3) or 
                    (self._write_head > 2 and self._write_head < self.buf_size - 3)):
                   
                    if self._read_head != 3 and self._write_head != 3:
                        rep += '                ...\n'

                    if self._write_head == self._read_head:
                        rep += (f" [{self._write_head:{self._width}d}] writer >>> "  +
                                 '[{}  ...  {}]'.format(
                                 self._format_array(self._buffer[self._write_head]), 
                                 self._format_array(self._buffer[self._write_head])) +
                                 "<<< reader [{self._read_head:{self._width}d}]\n")
                    else: 
                        writer_line = (f" [{self._write_head:{self._width}d}] writer >>> "  +
                                       '[{}  ...  {}]\n'.format(
                                           self._format_array(self._buffer[self._write_head][:3]), 
                                           self._format_array(self._buffer[self._write_head][-3:]))
                                       )
                        reader_line = (padding  +
                                       '  [{}  ...  {}]'.format(
                                           self._format_array(self._buffer[self._read_head][:3]), 
                                           self._format_array(self._buffer[self._read_head][-3:]))
                                        + f" <<< reader [{self._read_head:{self._width}d}]\n")
                        if self._read_head <= 2 or self._read_head >= self.buf_size - 3:
                            rep += writer_line
                        elif self._write_head <= 2 or self._write_head >= self.buf_size - 3:
                            rep += reader_line
                        else:
                            lines = ([writer_line, reader_line] 
                                     if self._write_head < self._read_head 
                                     else [reader_line, writer_line]
                                     )
                        
                            rep += lines[0] + '                ...\n' + lines[1]
                    
                    if self._read_head != self.buf_size - 4 and self._write_head != self.buf_size - 4:
                        rep += '                ...\n'
                else:
                    rep += '                ...\n'
                    

                for i, row in enumerate(self._buffer[-3:]):
                    i += self.buf_size-3
                    if i == self._write_head:
                        left = f"[{i:{self._width}d}] writer >>>"
                    else:
                        left = padding
                    if i == self._read_head:
                        right = f"<<< reader [{i:{self._width}d}]" 
                    else:
                        right = ""
                    rep += ' {} [{}  ...  {}] {}\n'.format(left,
                                                        self._format_array(row[:3]), 
                                                        self._format_array(row[-3:]),
                                                        right)
            else: 
                for i, row in enumerate(self._buffer):
                    if i == self._write_head:
                        left = f"[{i:{self._width}d}] writer >>>"
                    else:
                        left = padding
                    if i == self._read_head:
                        right = f"<<< reader [{i:{self._width}d}]" 
                    else:
                        right = ""
                    rep += '{} [{}  ...  {}] {}\n'.format(left,
                                                        self._format_array(row[:3]), 
                                                        self._format_array(row[-3:]),
                                                        right)
        else:
            if self.buf_size > 8:
                for i, row in enumerate(self._buffer[:3]):
                    if i == self._write_head:
                        left = f"[{i:{self._width}d}] writer >>>"
                    else:
                        left = padding
                    if i == self._read_head:
                        right = f"<<< reader [{i:{self._width}d}]" 
                    else:
                        right = ""
                    rep += ' {} [{}] {}\n'.format(left, self.format_array(row), right)
                rep += '                ...\n'
                for i, row in enumerate(self._buffer[-3:]):
                    i += self.buf_size-3
                    if i == self._write_head:
                        left = f"[{i:{self._width}d}] writer >>>"
                    else:
                        left = padding
                    if i == self._read_head:
                        right = f"<<< reader [{i:{self._width}d}]" 
                    else:
                        right = ""
                    rep += ' {} [{}] {}\n'.format(left, self.format_array(row), right)
            else:
                for i, row in enumerate(self._buffer):
                    if i == self._write_head:
                        left = f"[{i:{self._width}d}] writer >>>"
                    else:
                        left = padding
                    if i == self._read_head:
                        right = f"<<< reader [{i:{self._width}d}]" 
                    else:
                        right = ""
                    rep += ' {} [{}] {}\n'.format(left, self.format_array(row), right)

        return rep



