
from typing import Dict

from cocotb.queue import Queue
from cocotb.triggers import RisingEdge

from tb.monitor import Monitor

class TopMonitor (Monitor):
    def __init__(self, dut, variant):
        super().__init__(dut, variant)

    async def _run(self) -> None:
        while True:
            await RisingEdge(self.dut.core_clk)

            breakpoint()
            # print self.dut.c0_ddr4_s_axi_awvalid.