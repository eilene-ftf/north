import numpy as np
import nengo
import nengo_spa as spa

# if sub_req is false, sigout should pulse on each put and we should pop at regular intervals
# if pub_req is false, we should put at regular intervals and pulse sigout whenever something is popped
class RingBuffer(spa.Network):
    def __init__(self, buf_size, dim, pub=None, sub=None, writer_start=0, reader_start=-1, 
                 pub_req=True, sub_req=True, interval=None, t_pulse=0.2, t_delay=0.1, theta=0.3, 
                 vocab=None, pub_vocab=None, sub_vocab=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.buf_size = buf_size
        self._width = int(np.log10(self.buf_size-1)) + 1
        self._buffer = np.zeros((self.buf_size, dim))
        self._read_head = (self.buf_size + reader_start) % self.buf_size
        self._write_head = (self.buf_size + writer_start) % self.buf_size
        self._iter_flag = False
        self._reset_flag = True
        
        self.pub_vocab = pub_vocab
        self.sub_vocab = sub_vocab

        if vocab in kwargs:
            self.vocab = kwargs[vocab]
            if not self.pub_vocab:
                self.pub_vocab = vocab
            if not self.sub_vocab:
                self.sub_vocab = vocab
        else: 
            self.vocab = None

        assert self.label is not None and self.label.isalnum() and self.label[0].isalpha()

        self._pub_req = pub_req
        self._sub_req = sub_req
        self._interval = interval
        self._t_pulse = t_pulse
        self._t_delay = t_delay
        self._theta = theta


        if not (pub_req and sub_req) and not interval:
            raise ValueError("Must set an interval if either sub requests or pub requests are disabled")
        elif interval and interval < t_delay + t_pulse:
            raise ValueError(f"Interval must be at least as long as time to complete an operation, {t_delay + t_puls}")

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
                    
                    if t < 0.01 and not self._reset_flag:
                        self._reset()
                        self._reset_flag = True
                        print(self)
                    elif t > 0.01 and self._reset_flag:
                        self._reset_flag = False

                    from_pub = x[:self.dim]
                    pub_sigin = x[-1]

                    if state == 0 and pub_sigin > self._theta:
                        state = 1
                        self.put(from_pub)
                        stopwatch = t
                        print(self, 'put!')
                    elif state == 1 and t > stopwatch + self._t_delay:
                        state = 2
                    elif state == 2 and t > stopwatch + self._t_delay + self._t_pulse:
                        state = 0
                        stopwatch = 0

                    return state==2

                return write_state_machine

            def make_read_state_machine(sub_req):
                state = 0
                stopwatch = 0
                to_return = np.zeros(self.dim)
                def read_state_machine(t, x):
                    nonlocal state, stopwatch, to_return

                    sub_sigin = x[-1]

                    if state == 0 and sub_sigin > self._theta:
                        try:
                            to_return[:] = self.pop()
                            state = 1
                        except IndexError:
                            to_return[:] = 0
                            state = -1
                        stopwatch = t
                        print(self, "popped!")
                    elif state == 1 and t > stopwatch + self._t_delay:
                        state = 2
                    elif state == -1 and t > stopwatch + self._t_delay:
                        state = -2
                    elif np.abs(state) == 2 and t > stopwatch + self._t_delay + self._t_pulse:
                        state = 0
                        stopwatch = 0
                        #to_return[:] = 0

                    return np.concatenate([to_return, [state//2]])

                return read_state_machine



            read_controller = nengo.Node(size_in=1,
                                         size_out=self.dim+1,
                                         output=make_read_state_machine(sub_req),
                                         label="read_controller")

            write_controller = nengo.Node(size_in=self.dim+1,
                                         size_out=1,
                                         output=make_write_state_machine(pub_req),
                                         label="write_controller")

            nengo.Connection(self.pub.publisher.output, write_controller[:self.dim])
            nengo.Connection(read_controller[:self.dim], self.sub.subscriber.input)

            nengo.Connection(self.pub.publisher.sigin, write_controller[-1])
            nengo.Connection(self.sub.subscriber.sigin, read_controller)
            if self._pub_req:
                nengo.Connection(write_controller[-1], self.pub.publisher.sigout)
            else:
                nengo.Connection(read_controller[-1], self.pub.publisher.sigout)
            if self._sub_req:
                nengo.Connection(read_controller[-1], self.sub.subscriber.sigout)
            else:
                nengo.Connection(write_controller[-1], self.sub.subscriber.sigout)
       
    def _reset(self):
        self._buffer[:,:] = 0
        self._read_head = self.buf_size - 1
        self._write_head = 0
        self._iter_flag = False

    def _poll_node(self, interval):
        return nengo.Node(size_out=1, output=lambda t: t % interval < self._t_pulse)

    def _generate_pub(self):
        with self.pub:
            publabel = self.label + "_publisher"
            self.pub.publisher = spa.Network(label=publabel)
            publisher = self.pub.publisher
            setattr(self.pub, publabel, publisher)

            with publisher:
                if self.pub_vocab:
                    publisher.register = spa.State(self.pub_vocab, label="publisher.register")
                else:
                    publisher.register = spa.State(self.dim, label="publisher.register")
                publisher.input = nengo.Node(size_in=self.dim)
                publisher.output = nengo.Node(size_in=self.dim)
                if self._pub_req:
                    publisher.sigin = nengo.Node(size_in=1)
                else:
                    publisher.sigin = self._poll_node(self._interval)
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
                if self.sub_vocab:
                    subscriber.register = spa.State(self.sub_vocab, label="subscriber.register")
                else:
                    subscriber.register = spa.State(self.dim, label="subscriber.register")
                subscriber.input = nengo.Node(size_in=self.dim)
                subscriber.output = nengo.Node(size_in=self.dim)
                if self._sub_req:
                    subscriber.sigin = nengo.Node(size_in=1)
                else:
                    subscriber.sigin = self._poll_node(self._interval)
                subscriber.sigout = nengo.Node(size_in=1)

                nengo.Connection(subscriber.input, subscriber.register.input)
                nengo.Connection(subscriber.register.output, subscriber.output)


    def __iter__(self):
        return self

    def __next__(self):
        if self._iter_flag:
            raise StopIteration
        return self.pop()

    def put(self, item):
        if isinstance(item, spa.SemanticPointer):
            item = item.v
        self._buffer[self._write_head, :] = item[:]
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

    def put_list(self, lis):
        for item in lis:
            self.put(item)

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
                                     self._format_array(self._buffer[self._write_head][:3]), 
                                     self._format_array(self._buffer[self._write_head][-3:])) +
                                 f" <<< reader [{self._read_head:{self._width}d}]\n")
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



