import os

from tb.utils.common import NodePrecision, AggregationFunction
from tb.variant import Variant
from sdk.pcie_manager import PCIeManager

class Ample_Driver:
    def __init__(self,variant,sim= False):

      # self.variant = Variant(message_channel_count, precision_count, aggregation_buffer_slots)
      self.layer_config_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/layer_config.json"
      self.regbank_path = os.environ.get("WORKAREA") + "/hw/build/regbanks"

      if not sim:
        self.device_manager = PCIeManager()

      self.nodeslot_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/nodeslot_programming.txt"
      self.memory_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/memory.mem"

      #XDMA AXI Offsets
      self.ram_c0_offset = 0x0000_0004_0000_0000
      self.ram_c1_offset = 0x0000_0000_0000_0000

      #XDMA AXI-L Offsets
      self.led_reg_offset = 0x4000_0000
      self.regbank_offset = 0x0000_0000


      self.nsb_regbank = {}
      self.age_regbank = {}
      self.prefetcher_regbank = {}
      self.fte_regbank = {}
      self.layers = {}



    def execute(self):

      #Remove biases
      for layer_idx, layer in enumerate(self.layers):

        print(f"Executing layer {layer_idx}")
        print(f"Layer: {layer}")

        self.request_weights_fetch(precision=NodePrecision.FLOAT_32)
        print("Weights fetch done.")

        self.device_manager.write_memory(self.nsb_regs["graph_config_node_count"], self.layers[layer_idx]['nodeslot_count'])
       
        #Temp TODO program nodeslot start address for layer
        # print(f"Layer {layer_idx} nodeslot start address: {test.layers[layer_idx]['nodeslot_start_address']}")
        self.device_manager.write_memory(self.nsb_regs["ctrl_start_nodeslot_fetch_start_addr"], self.layers[layer_idx]['nodeslot_start_address'])
        self.device_manager.write_memory(self.nsb_regs["concat_width"], self.layers[layer_idx]['concat_width'])

        self.device_manager.write_memory(self.nsb_regs["ctrl_start_nodeslot_fetch"], 1)

        self.wait_done_ack(
            done_reg = self.driver.nsb_regs["ctrl_start_nodeslot_fetch_done"],
            ack_reg = self.driver.nsb_regs["ctrl_start_nodeslot_fetch_done_ack"],
            tries = 10000
        )

        #TODO log time start
        print("Nodeslot fetching done, waiting for nodeslots to be flushed.")
        #Multi thread, interupt?
        self.flush_nodeslots()
        #TODO log time end


    def load_nodeslot_file(self):
      self.device_manager.write_file(self.nodeslot_file, self.ram_c1_offset)

    def load_memory_file(self):
      self.device_manager.write_file(self.memory_file, self.ram_c0_offset)

    def load_layer_config(self):
      self.dut._log.debug("Loading layer configuration")
      with open(self.layer_config_file) as f:
          data = json.load(f)
      self.global_config = data["global_config"]
      self.layers = data["layers"]
      return data["layers"]

    def load_regbank(self, regbank):
      json_path = os.path.join(self.regbank_path, regbank, regbank + "_regs.json")
      self.dut._log.debug("Loading %s from %s", regbank, json_path)
      with open(json_path) as f:
          data = json.load(f)
      return data


    
    def load_regbanks(self):
        self.dut._log.debug("Loading register banks.")

        nsb_regmap = self.load_regbank("node_scoreboard_regbank")["registerMap"]
        prefetcher_regmap = self.load_regbank("prefetcher_regbank")["registerMap"]
        age_regmap = self.load_regbank("aggregation_engine_regbank")["registerMap"]
        fte_regmap = self.load_regbank("feature_transformation_engine_regbank")["registerMap"]

        self.nsb_regbank = {register["name"]: register for register in nsb_regmap["registers"]}
        self.prefetcher_regbank = {register["name"]: register for register in prefetcher_regmap["registers"]}
        self.age_regbank = {register["name"]: register for register in age_regmap["registers"]}
        self.fte_regbank = {register["name"]: register for register in fte_regmap["registers"]}

        self.driver.nsb_regs = {register["name"]: nsb_regmap["baseAddress"] + register["addressOffset"] for register in self.nsb_regbank.values()}
        self.driver.prefetcher_regs = {register["name"]: prefetcher_regmap["baseAddress"] + register["addressOffset"] for register in self.prefetcher_regbank.values()}
        self.driver.age_regs = {register["name"]: age_regmap["baseAddress"] + register["addressOffset"] for register in self.age_regbank.values()}
        self.driver.fte_regs = {register["name"]: fte_regmap["baseAddress"] + register["addressOffset"] for register in self.fte_regbank.values()}

    def program_layer_config(self):
      # Prefetcher register bank
      print("Programming prefetcher register bank layer configuration.")
      print()
      self.device_manager.write_to_memory(self.prefetcher_regs["layer_config_in_features"], layer["in_feature_count"])
      self.device_manager.write_to_memory(self.prefetcher_regs["layer_config_out_features"], layer["out_feature_count"])
      
      # Addresses
      self.device_manager.write_to_memory(self.prefetcher_regs["layer_config_adjacency_list_address_lsb"], layer["adjacency_list_address"])

      self.device_manager.write_to_memory(self.prefetcher_regs["layer_config_weights_address_lsb"], layer["weights_address"])
      self.device_manager.write_to_memory(self.prefetcher_regs["layer_config_in_messages_address_lsb"], layer["in_messages_address"])

      # AGE register bank
      self.dut._log.debug("Programming AGE register bank layer configuration.")
      self.device_manager.write_to_memory(self.age_regs["layer_config_in_features"], layer["in_feature_count"])
      self.device_manager.write_to_memory(self.age_regs["layer_config_out_features"], layer["out_feature_count"])

      # FTE register bank

      self.dut._log.debug("Programming FTE register bank layer configuration.")
      self.device_manager.write_to_memory(self.fte_regs["layer_config_in_features"], layer["in_feature_count"])
      self.device_manager.write_to_memory(self.fte_regs["layer_config_out_features"], layer["out_feature_count"])

      #LSB and MSB bug potential TODO Split into MSB and LSB
      self.device_manager.write_to_memory(self.fte_regs["layer_config_out_features_address_lsb"], layer["out_messages_address"])

      # NSB register bank
      self.dut._log.debug("Programming NSB register bank layer configuration.")
      self.device_manager.write_to_memory(self.nsb_regs["layer_config_in_features"], layer["in_feature_count"])
      self.device_manager.write_to_memory(self.nsb_regs["layer_config_out_features"], layer["out_feature_count"])
      # Addresses
      self.device_manager.write_to_memory(self.nsb_regs["layer_config_adjacency_list_address_lsb"], layer["adjacency_list_address"])
      self.device_manager.write_to_memory(self.nsb_regs["layer_config_weights_address_lsb"], layer["weights_address"])
      # Wait counts
      self.device_manager.write_to_memory(self.nsb_regs["NSB_CONFIG_AGGREGATION_WAIT_COUNT"], layer["aggregation_wait_count"])
      #Aggregate Enable
      self.device_manager.write_to_memory(self.nsb_regs["layer_config_aggregate_enable"], layer["aggregate_enable"])

      # Set config valid
      self.device_manager.write_to_memory(self.nsb_regs["layer_config_valid"], 1)


    def request_weights_fetch(self, precision=NodePrecision["FLOAT_32"]):
      print("Requesting weights fetch for precision %s.", precision.name)
      self.device_manager.write_to_memory(self.nsb_regs["ctrl_fetch_layer_weights_precision"], precision.value)
      self.device_manager.write_to_memory(self.nsb_regs["CTRL_FETCH_LAYER_WEIGHTS"], 1)

      self.wait_done_ack(
          done_reg = self.nsb_regs["CTRL_FETCH_LAYER_WEIGHTS_DONE"],
          ack_reg = self.nsb_regs["CTRL_FETCH_LAYER_WEIGHTS_DONE_ACK"],
          tries = 100000
      )

    def wait_done_ack(self, done_reg, ack_reg, tries=100):
      done = False
      for _ in range(tries):
          done = self.device_manager.read_from_memory(done_reg)
          if (done):
              # Weights fetch done, write to ACK
              print(f"{done_reg} register is asserted")
              self.device_manager.write_to_memory(ack_reg, 1)
              break
          #TODO add suitable delay
          # await delay(self.dut.regbank_clk, 10)
      
      if (not done):
          self.dut._log.info(f"Tried reading {done_reg} register {tries} times, but still not done. Simulation hung?")

    def flush_nodeslots(self):
        # Wait for work to finish
        print("Waiting for nodeslots to be empty.")
        while(True):
            # Build free mask
            free_mask = ''
            for i in range(0, int(self.nodeslot_count/32)):
                empty_mask = self.device_manager.read_memory(self.driver.nsb_regs["status_nodeslots_empty_mask_" + str(i)])
                free_mask = empty_mask.binstr + free_mask
            
            # self.dut._log.debug("Free nodeslots: %s", free_mask)

            if (free_mask == "1" * self.nodeslot_count):
                break
            
            # await delay(self.dut.regbank_clk, 10)
            #TODO add suitable delay