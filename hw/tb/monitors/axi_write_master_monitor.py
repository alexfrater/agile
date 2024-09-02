import cocotb
from cocotb.triggers import RisingEdge
from cocotb.log import SimLog
from cocotb.result import TestFailure
import numpy as np

import struct

import torch

from cocotb.binary import BinaryValue   
import logging

from torch import tensor


class AXIWriteMasterMonitor:
    def __init__(self, dut, clk, req_valid, req_ready, start_address, req_len, data_valid, data, pop, resp_valid, resp_ready, tolerance = 1e-1, log_level=logging.INFO):
        self.dut = dut
        self.clk = clk
        self.req_valid = req_valid
        self.req_ready = req_ready
        self.start_address = start_address
        self.req_len = req_len
        self.data_valid = data_valid
        self.data = data
        self.pop = pop
        self.resp_valid = resp_valid
        self.resp_ready = resp_ready
        self.transactions = {}  # Tracking ongoing transactions
        # self.log = SimLog("cocotb.AXIWriteMasterMonitor")
        # self.log.setLevel(log_level)  # Set to the desired level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        self.tolerance = tolerance
        self.expected = {}  # Tracking ongoing transactions
        self.running = False
        self.expected_layer_features_by_address = {}

    def kill(self):
        self.dut._log.debug("Killing monitor")
        self._running = False
        if self._thread:
            self._thread.kill()
            self._thread = None


    async def monitor_write_transactions(self):
        while True:
            if self.running == False:
                return
            
            await RisingEdge(self.clk)
            # Write Request
            if self.req_valid.value and self.req_ready.value: #Check not happening twice
                start_addr = int(self.start_address.value)
                length = int(self.req_len.value)
                current_transaction = {
                    'start_address': start_addr,
                    'data': [],
                    'expected_length': length+1
                }
                self.dut._log.debug(f"Transaction {current_transaction}")

            # Data
            if self.data_valid.value and self.pop.value and current_transaction:
                data_chunk = hex(self.data.value)
                current_transaction['data'].append(self.hex_to_floats(data_chunk))


            # Response
            if self.resp_valid.value and self.resp_ready.value and current_transaction: #Rmeove current_transaction and fix RTL to be high for single clock cycle - check if different clock e.g 100Mhz - 2 cycles at 200MHz
                assert len(current_transaction['data']) == current_transaction['expected_length'], f"Transaction data length mismatch at address {current_transaction['start_address']}"

 
                self.dut._log.debug("Getting node")
                expected_node = self.get_node_by_address(current_transaction['start_address'])
                if expected_node: #Change to assertion
                    self.dut._log.debug("--------------------")
                    self.dut._log.debug("")
                    self.dut._log.debug(f"Node found: {expected_node['node_id']}, Address: {expected_node['address']}")
                    # self.dut._log.debug(f"Actual Address: {current_transaction['start_address']}")

                    # Check
                    self.dut._log.debug(f"Data expected {expected_node['data']}")
                    current_transaction['data'] = tensor([item for sublist in current_transaction['data'] for item in sublist[::-1]])
                    self.dut._log.debug(f"Data gotten {current_transaction['data']}")
                    assert current_transaction['data'].shape == expected_node['data'].shape, f"Data size mismatch for address {current_transaction['start_address']}"


                    #TODO reinstate this when edge aggregation is working
                    assert torch.allclose(current_transaction['data'], expected_node['data'], atol=self.tolerance), \
                        f"Data mismatch for node {expected_node['node_id']} address {current_transaction['start_address']}"

                    self.dut._log.debug(f"Data and address correctly matched for node: {expected_node['node_id']}")
                    self.dut._log.debug(" ")
                    self.dut._log.debug("--------------------")
                else:
                    self.dut._log.warning(f"No node found with address {current_transaction['start_address']}")
                
                    
                if current_transaction:
                    del current_transaction['start_address']
                    current_transaction = None


    def load_layer_features(self, nodeslot_programming,layer_features,layer_config,global_config,layer_idx):
        self.dut._log.debug("Loading Layer Features")
        #Single model models - TODO 
        self.dut._log.debug(f"{layer_config['name']}")
        if len(nodeslot_programming) ==1:
            nodeslot_group_programming = nodeslot_programming[0]
        else:
        #Need to load in correct nodeslot group
            #Change to submodel id
            nodeslot_group = layer_config['sub_model_id']

            #Use new nodelsot group for each layer? TODO change for edge node groups
            nodeslot_group_programming = nodeslot_programming[nodeslot_group]


        if layer_config['edge_node']:
            nodeslots = nodeslot_group_programming[1]
            edge_offset = global_config['node_count'] #TODO Fix
        else:
            edge_offset = 0
            nodeslots = nodeslot_group_programming[0]


        self.expected_layer_features_by_address = {}
        layer_out_message_offset = layer_config['out_messages_address']



        for nodeslot in nodeslots:
            node_id = nodeslot['node_id']

            data = layer_features[node_id-(edge_offset)] #Remove edge count offset to look at results

            out_messages_address_lsb = nodeslot['out_messages_address_lsb'] + layer_out_message_offset
            axi_write_master_address =int(out_messages_address_lsb)
            node_dict = {
                'node_id': node_id,
                'address': axi_write_master_address, 
                'data': data
            }

            self.expected_layer_features_by_address[axi_write_master_address] = node_dict

        self.dut._log.debug("Expected Data Indexed by Address:")
        for address, node in self.expected_layer_features_by_address.items():
            self.dut._log.debug(f"Address: {address}, Node: {node['node_id']}")
            self.dut._log.debug(f"Data: {node['data']}")


    def get_node_by_address(self,address):
        return self.expected_layer_features_by_address.pop(address, None)

      
    def empty_expected_layer_features(self):
        return (self.expected_layer_features_by_address == {})

    def expected_layer_features(self):
        return self.expected_layer_features_by_address
    


    def hex_to_floats(self,hex_string):
        if hex_string.startswith('0x'):
            hex_string = hex_string[2:]
        
        hex_string = hex_string.replace(' ', '').replace('\n', '')
        
        if any(c not in '0123456789abcdefABCDEF' for c in hex_string):
            raise ValueError("Non-hexadecimal character found in input string")
        
        if not hex_string:
            return []
        
        if len(hex_string) % 8 != 0:
            hex_string = hex_string.ljust((len(hex_string) + 7) // 8 * 8, '0')
        
        byte_data = bytes.fromhex(hex_string)
        
        num_floats = len(byte_data) // 4
        
        floats = struct.unpack('>' + 'f' * num_floats, byte_data)
        
        return floats
    


    