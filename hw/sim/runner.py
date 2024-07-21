import cocotb
import sys
sys.path.append('/home/aw1223/ip/agile')
from tb.runners.graph_test_runner import graph_test_runner
from cocotb.log import SimLog

import logging


@cocotb.test()
async def graph_test(dut):
    runner_log = SimLog("cocotb.runner")

    await graph_test_runner(dut)
    runner_log.info(f"DONE")

    # await my_loop_test(dut)
    
