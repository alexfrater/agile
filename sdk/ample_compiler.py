
import os 
import json
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data


from tb.variant import Variant
from sdk.model_tracer import ModelTracer
from sdk.initialization_manager import InitManager
from sdk.trained_graph import TrainedGraph
from .utilities import dump_byte_list, binary_to_hex


class AmpleCompiler():
  def __init__(self,sim = False):
    # self.model = None
    self.variant = Variant()
    self.sim = sim

    self.model_tracer = ModelTracer()

    self.mem_append= False

    self.base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config/"
    self.layer_config_file = self.base_path + "layer_config.json"
    self.nodeslot_file = self.base_path + "nodeslot_programming.json"
    self.nodeslot_programming_group_start_address = []
    self.nodeslot_mem_dump_file = self.base_path  + "nodeslots.mem"

  #TODO make it so that inputs can be passed in aribtary order
  def compile(
        self,
        model,
        data,
        base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config",
        precision = 'FLOAT_32',
        reduce = False,
        random = True,
        trained = False,
        plot = True,
        trace_mode = 'hooks'
    ):

        self.mem_append= False
        self.nodeslot_programming = []

        self.model_trace = self.model_tracer.trace_model(model, data)
        if plot:
            # print(self.model_trace)
            self.model_tracer.plot_model()

        all_outputs = set()
        for sub_module_name, sub_module_dict in self.model_trace.items():
            all_outputs.update(sub_module_dict['output_names'])

        #Ensure the modules are in the correct order to compile (i.e., input addresses are calculated before module)
        # self.model_trace = self.reorder_modules(self.model_trace) 
        
        # Identify external inputs
        external_inputs = set()
        for sub_module_name,sub_module_dict in self.model_trace.items():
            for input_name in sub_module_dict['input_names']:
                if input_name not in all_outputs:
                    external_inputs.add(input_name)
        
        self.memory_ptr = 0 #keep track of memory address

        external_inputs_dict = {element: None for element in external_inputs}
        external_inputs_dict = dict(sorted(external_inputs_dict.items()))
        print('Ensure inputs match the following list', external_inputs_dict)
        #Map external inputs to dict #TODO find a way to do withut providing sorted inputs - pass in inputs arbitraly and then map to names
        for key, value in zip(external_inputs_dict.keys(), data):
            external_inputs_dict[key] = value

         
        edge_index_external_inputs_dict = {key: value for key, value in external_inputs_dict.items() if isinstance(value, torch.Tensor) and value.shape[0] == 2}


        #List of number of edges in edge index tensors
        edge_num_list = []
        for key, value in edge_index_external_inputs_dict.items():
            edge_num_list.append(value.shape[1])
        #Find edge attrbiutes by checking if number of inputs is same as number of edges - TODO make more robust
        edge_attr_external_inputs_dict = {key: value for key, value in external_inputs_dict.items() if isinstance(value, torch.Tensor) and value.shape[0] in edge_num_list}

        #Find number of inputs is same as number of nodes
        node_feature_external_inputs_dict = {key: value for key, value in external_inputs_dict.items()
                                             if isinstance(value, torch.Tensor) and value.shape[0] != 2 and key not in edge_attr_external_inputs_dict
                                             }
        for sub_model_id, (sub_module_name, sub_module_dict) in enumerate(self.model_trace.items()):
            input_names = sub_module_dict['input_names']
            module_type = sub_module_dict['module_type']
            assert module_type in self.model_tracer.model_map, f"Module type {module_type} not supported."
           
            edge_attr_messages_addr = None
            in_message_addr = None
            #TODO merge external and internal inputs
            ###DATA###
            #Type 0 - External input - may have external edge input
            if any(name in node_feature_external_inputs_dict or name in edge_attr_external_inputs_dict for name in input_names):

                dataset = Data()

                for name in input_names:
                    if name in node_feature_external_inputs_dict:       
                        x = list(node_feature_external_inputs_dict[name])
                        #TODO Change to acccept multpile X inputs e.g grid,mesh features
                        if dataset.x is None:
                            dataset.x = x
                            dataset.num_nodes = len(dataset.x)
                        else:
                            dataset.x = [torch.cat((dataset.x, x), dim=1)]
                            dataset.num_nodes = dataset.num_nodes + len(x)
                        sub_module_dict['num_nodes'] = dataset.num_nodes

                    elif name in edge_index_external_inputs_dict:

                        dataset.edge_index = edge_index_external_inputs_dict[name]

                    elif name in edge_attr_external_inputs_dict:
                        dataset.edge_attr = edge_attr_external_inputs_dict[name]
                        # sub_module_dict['num_nodes'] = len(dataset.edge_attr)

                #TODO replace this
                if dataset.edge_index is  None and dataset.x is None: #check if edge attr embedder #TODO make more robust
                    dataset.num_nodes = len(dataset.edge_attr)
                    dataset.x = dataset.edge_attr #Make x input the edge attr as it is treating them as nodes to embed
                    dataset.edge_attr = None 
                    sub_module_dict['num_nodes'] = dataset.num_nodes
            #Type 0 - Interal input - may have external edge index
            else:
                dataset = Data()
                dataset.x = None

                #Edge index assingment - Always external
                for name in input_names:
                    if name in edge_index_external_inputs_dict:
                        dataset.edge_attr = True #TODO check if edge attr is present - use model name? - use edge attr internal dict
                        dataset.edge_index = edge_index_external_inputs_dict[name]
    
                #Edge and Node Features assignment
                input_nodes = []
                for item_name, item_data in self.model_trace.items():
                    if any(input_id in item_data['output_names'] for input_id in input_names):
                        num_items =  self.model_trace[item_name]['num_nodes']
                        #TODO make edge atttribute test more robust - if num edges is same as num nodes it will fail
                        #Edge attribute
                        
                        if num_items in edge_num_list: #Edge attribute
                            # dataset.num_nodes = self.model_trace[item_name]['num_nodes']
                            # sub_module_dict['num_nodes'] = dataset.num_nodes
                            edge_attr_messages_addr = self.model_trace[item_name]['out_addr']

                        else:
                            dataset.num_nodes = self.model_trace[item_name]['num_nodes']
                            sub_module_dict['num_nodes'] = dataset.num_nodes
                            in_message_addr = self.model_trace[item_name]['out_addr']

                        input_nodes.append(item_name)
            print(f"Sub model {sub_module_name} has inputs from {input_names}")
            self.model_trace[sub_module_name]['out_addr'] = self.initialize_sub_model_memory(
                                                                                        sub_model = sub_module_dict['module'],
                                                                                        dataset = dataset,
                                                                                        feature_count=32,
                                                                                        sub_model_id = sub_model_id,
                                                                                        in_message_addr=in_message_addr,
                                                                                        edge_attr_messages_addr = edge_attr_messages_addr,
                                                                                        )
        
        self.dump_nodeslot_programming_multi_model()
        self.add_nodeslot_addresses()

        if self.sim:
            self.save_model(model,data)
            self.save_graph(data)

  def initialize_sub_model_memory( 
        self,
        sub_model=None,
        dataset = None,
        feature_count=32,
        in_message_addr = None,
        edge_attr_messages_addr = None,
        sub_model_id = None,
        precision = 'FLOAT_32',
        reduce =False,
        random = True,
        trained = False,
        date = True
    ):

        d_type = self.get_dtype(precision)
        self.graph = TrainedGraph(dataset,feature_count)
        self.init_manager = InitManager(self.graph, sub_model,self.memory_ptr, base_path=self.base_path,sub_model_id=sub_model_id)
        if dataset.x is not None:
            self.init_manager.trained_graph.load_embeddings()
         
        self.init_manager.map_memory(in_messages_addr = in_message_addr,edge_attr_messages_addr=edge_attr_messages_addr) 
        self.init_manager.dump_memory(self.mem_append)
        
        self.nodeslot_programming.append(self.init_manager.return_nodeslot_programming())
        self.memory_ptr,out_messages_address = self.init_manager.dump_layer_config(self.mem_append)

        if self.mem_append == False: #TODO 
            self.mem_append = True
        return out_messages_address

  def get_dtype(self,precision):
    if precision == 'FLOAT_32':
        dtype = torch.float32
    elif precision == 'FIXED_16':
        dtype = torch.float16
    elif precision == 'FIXED_8':
        dtype = torch.uint8
    elif precision == 'FIXED_4':
        dtype = torch.uint8  # PyTorch does not support uint4, using uint8 as a placeholder
    else:
        dtype = torch.float32

    return dtype


  def dump_nodeslot_programming_multi_model(self,append_mode=False):
    mode = 'a' if append_mode else 'w'

    with open(self.nodeslot_file,'w') as file:
        json.dump(self.nodeslot_programming, file, indent=4)

    nodeslot_memory_pointer = 0
    nodeslot_byte_list = []

    for group, nodeslot_group in enumerate(self.nodeslot_programming):
        node_ptr = 0
        edge_ptr = 0

        node_group = nodeslot_group[0]

        node_ptr = nodeslot_memory_pointer
        subgroup_byte_list, nmh_length = self.generate_nodeslots_mem(node_group)
        nodeslot_byte_list += subgroup_byte_list
        nodeslot_memory_pointer += nmh_length

        if len(nodeslot_group)>1:            
            edge_group = nodeslot_group[1]
            edge_ptr = nodeslot_memory_pointer
            subgroup_byte_list, nmh_length = self.generate_nodeslots_mem(edge_group)
            nodeslot_byte_list += subgroup_byte_list
            nodeslot_memory_pointer += nmh_length

        self.nodeslot_programming_group_start_address.append((node_ptr,edge_ptr))
      

    dump_byte_list(nodeslot_byte_list, self.nodeslot_mem_dump_file, append_mode)


  def generate_nodeslots_mem(self,nodeslot_group):
    node_groups = np.array(nodeslot_group)
    
    node_groups = np.pad(
                            node_groups, 
                            (0, 8 - node_groups.shape[0] % 8), 
                            "constant", 
                            constant_values=None
                        ).reshape(-1, 8)
    
    nodeslot_mem_hex = []

    for group_idx,group in tqdm(enumerate(node_groups)):
        # logging.info(f"Generating nodeslot group {group_idx} memory.")

        assert(len(group) == 8)
        str_lst = []
        for nodeslot in group:
            if (nodeslot is None) or (nodeslot['neighbour_count'] == 0):
                str_lst.append("0"*64)
                continue
            str = ""
            str = f"{nodeslot['neighbour_count']:20b}{nodeslot['node_id']:20b}" + str
            str = "00" + str
            str = "1" + "0"*21 + str
            str = str.replace(" ", "0")
            str_lst.append(str)
        
        str = "".join(str_lst[::-1])
        hex = binary_to_hex(str).zfill(128)
        nodeslot_mem_hex += "".join(hex)
    
    assert (len(nodeslot_mem_hex) % 2 == 0)
    nmh = [nodeslot_mem_hex[i] + nodeslot_mem_hex[i+1] for i in range(0, len(nodeslot_mem_hex), 2)]

    nodeslot_mem_len = len(nodeslot_mem_hex)//2 #byte indexed
    return nmh,nodeslot_mem_len

  def add_nodeslot_addresses(self):
    with open(self.layer_config_file, 'r') as file:
        data = json.load(file)

    for i, layer in enumerate(data['layers']):

        sub_module_nodeslot_grp = self.nodeslot_programming_group_start_address[layer['sub_model_id']]
        if layer['edge_node']:
            layer['nodeslot_start_address'] = sub_module_nodeslot_grp[1]
        else:
            layer['nodeslot_start_address'] = sub_module_nodeslot_grp[0]

    with open(self.layer_config_file, 'w') as file:
        json.dump(data, file, indent=4)

  #Save JIT model for testbench
  def save_model(self,model,inputs):
    model.eval()
    jit_model = torch.jit.trace(model, tuple(inputs))
    torch.jit.save(jit_model, self.base_path + 'model.pt')

    return jit_model

    
  #Save graph for testbench
  def save_graph(self,inputs):
    input_data =inputs

    torch.save({
        'input_data': input_data
    }, self.base_path + 'graph.pth')
