import numpy as np

from cocotb.binary import BinaryValue
from cocotb.triggers import *

from queue import Queue

import cocotb
from cocotb.log import SimLog
from cocotb.triggers import RisingEdge
from cocotb.result import TestFailure


class Monitor:
    """Simplified version of cocotb_bus.monitors.Monitor"""

    def __init__(self, clk, check=True, name=None):
        self.clk = clk
        self.recv_queue = Queue()
        self.exp_queue = Queue()
        self.check = check
        self.name = name

        if not hasattr(self, "log"):
            self.log = SimLog(
                "cocotb.monitor.%s" % (type(self).__qualname__)
                if self.name == None
                else self.name
            )

        self._thread = cocotb.scheduler.add(self._recv_thread())

    def kill(self):
        if self._thread:
            self._thread.kill()
            self._thread = None

    def expect(self, transaction):
        self.exp_queue.put(transaction)

    async def _recv_thread(self):
        while True:
            await RisingEdge(self.clk)
            if self._trigger():
                tr = self._recv()
                self.log.debug(f"Observed output beat {tr}")
                # self.recv_queue.put(tr)

                # if self.exp_queue.empty():
                #     raise TestFailure(
                #         "\nGot \n%s,\nbut we did not expect anything."
                #         % self.recv_queue.get()
                #     )

                # self._check(self.recv_queue.get(), self.exp_queue.get())

    def _trigger(self):
        raise NotImplementedError()

    def _recv(self):
        raise NotImplementedError()

    def _check(self, got, exp):
        raise NotImplementedError()

    def clear(self):
        self.send_queue = Queue()

    def load_monitor(self, tensor):
        for beat in tensor:
            self.log.debug(f"Expecting output beat {beat}")
            self.expect(beat)



class StreamMonitor(Monitor):
    def __init__(self, clk, data, valid, ready, check=True, name=None):
        super().__init__(clk, check=check, name=name)
        self.clk = clk
        self.data = data
        self.valid = valid
        self.ready = ready
        self.check = check
        self.name = name
        self.start = False

    def _trigger(self):
        if self.start == True:
            if self.ready.value == 1:
                # if self.valid.value == 1:
                    return True

    def _recv(self):
        if type(self.data.value) == list:
            return [int(x) for x in self.data.value]
        elif type(self.data.value) == BinaryValue:
            return int(self.data.value)

    def _check(self, got, exp):

        if self.check:
            if not np.equal(got, exp).all():
                self.log.error(
                    "%s: \nGot \n%s, \nExpected \n%s"
                    % (
                        self.name if self.name != None else "Unnamed StreamMonitor",
                        got,
                        exp,
                    )
                )
                raise TestFailure("\nGot \n%s, \nExpected \n%s" % (got, exp))


class StreamMonitorFloat(StreamMonitor):
    def __init__(self, clk, data, valid, ready, data_width, frac_width, check=True):
        super().__init__(clk, data, valid, ready, check)
        self.data_width = data_width
        self.frac_width = frac_width

    def _check(self, got, exp):
        if self.check:
            float_got = [x * 2**-self.frac_width for x in got]
            float_exp = [x * 2**-self.frac_width for x in exp]
            if not np.isclose(float_got, float_exp, atol=2**-self.frac_width).all():
                # raise TestFailure("\nGot \n%s, \nExpected \n%s" % (got, exp))
                raise TestFailure(
                    f"\nGot int \n{got}, \nExpected int \n{exp} \nGot float \n{float_got}, \nExpected float \n{float_exp}"
                )
            