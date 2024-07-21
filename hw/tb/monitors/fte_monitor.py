
from typing import Dict

from cocotb.queue import Queue
from cocotb.triggers import RisingEdge

from tb.monitor import Monitor

class FTE_Monitor (Monitor):
    def __init__(self, dut, variant):
        super().__init__(dut, variant)

        self.nsb_responses = Queue[Dict[str, int]]()
        self.weight_channel_reqs = [Queue[Dict[str, int]]() for _ in range(self.precision_count)] #Why not in range(self.message_channel_count)?
        self.axi_write_master_data = Queue[Dict[str, int]]()
        # self.start = False

    async def _run(self) -> None:
        while True:
            await RisingEdge(self.dut.core_clk)

            # if (self.start == True):

            # TFE AXI Writes
            if (self.dut.axi_write_master_data_valid.value == 1):
                data = {"payload": self.dut.axi_write_master_data, "address":self.dut.axi_write_master_req_start_address}
                self._sample(data, self.axi_write_master_data)
                self.dut._log.info("Observed FTE Write: %s, Address: %s", data["payload"].value.integer, data["address"].value.integer)


            # # NSB responses
            # if (self.dut.nsb_fte_resp_valid):
            #     data = {
            #         "nodeslots": self.dut.nsb_fte_resp.nodeslots,
            #         "precision": self.dut.nsb_fte_resp.precision
            #     }
            #     self._sample(data, self.nsb_responses)
            #     self.dut._log.info("Observed response to NSB for nodeslots: %s, precision: %s", data["nodeslots"].value, data["precision"].value)

            # # Weight Channel requests
            # for wc in range(self.message_channel_count):
            #     if (self.dut.weight_channel_req_valid[wc].value and self.dut.weight_channel_req_ready[wc].value):
            #         data = {
            #             "in_features": self.dut.weight_channel_req[wc].in_features,
            #             "out_features": self.dut.weight_channel_req[wc].out_features
            #         }
            #         _ = self._sample(data, self.weight_channel_reqs[wc])
            #         self.dut._log.info("Observed Weight Channel request for Nodeslot: %s, Fetch Tag: ", data["nodeslot"].value, data["fetch_tag"].value)

            # TFE AXI Writes

            # for prec in range(self.precision_count):
            #     if (self.dut.axi_write_master_resp_valid[prec]):
            #         data = {"payload": self.dut.axi_write_master_data[prec], "address":self.dut.axi_write_master_req_start_address[prec]}
            #         self._sample(data, self.axi_write_master_data[prec])
            #         self.dut._log.info("Observed FTE Write: %s, Address: ", data["payload"].value, data["address"].value)


       
           

# import csv
# import cocotb
# from cocotb.queue import Queue
# from cocotb.triggers import RisingEdge
# from tb.monitor import Monitor

# class FTE_Monitor(Monitor):
#     def __init__(self, dut, variant, feature_file):
#         super().__init__(dut, variant)
#         self.feature_file = feature_file
#         self.features_from_file = Queue()
#         self.axi_write_master_data = Queue()

#     async def load_features(self):
#         try:
#             with open(self.feature_file, 'r') as file:
#                 csv_reader = csv.reader(file)
#                 for row in csv_reader:
#                     features = [float(num) for num in row]
#                     await self.features_from_file.put(features)
#         except Exception as e:
#             self.dut._log.error(f"Error reading feature file: {e}")

#     async def _run(self):
#         # Start loading features from the CSV file
#         cocotb.start_soon(self.load_features())

#         while True:
#             await RisingEdge(self.dut.core_clk)

#             # Process AXI Writes
#             if self.dut.axi_write_master_data_valid.value == 1:
#                 data = {
#                     "payload": self.dut.axi_write_master_data,
#                     "address": self.dut.axi_write_master_req_start_address
#                 }
#                 self._sample(data, self.axi_write_master_data)
#                 self.dut._log.info(f"Observed FTE Write: {data['payload'].value.integer}, Address: {data['address'].value.integer}")

#                 # Compare to CSV features
#                 if not self.features_from_file.empty():
#                     expected_features = await self.features_from_file.get()
#                     # Compare features from file and AXI data (simplified comparison, adjust as needed)
#                     if not all(a == b for a, b in zip(expected_features, data['payload'].value.integer)):
#                         self.dut._log.error(f"Feature mismatch: Expected {expected_features}, got {data['payload'].value.integer}")

#     def _sample(self, data, queue):
#         """
#         Samples the data signals and builds a transaction object
#         """
#         queue.put_nowait(data)  # Using data directly for simplicity in this example
#         return data

# # # Usage
# # # Instantiate the monitor with the path to the CSV file containing the features
# # fte_monitor = FTE_Monitor(dut, variant, 'path_to_features.csv')
# # fte_monitor.start()
