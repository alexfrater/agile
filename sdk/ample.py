import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from graphviz import Digraph
from IPython.display import Image, display
import torch.fx as fx

from sdk.initialization_manager import InitManager
from sdk.benchmarking_manager import BenchmarkingManager
from sdk.models.models import GCN_Model, GAT_Model, GraphSAGE_Model, GIN_Model, GCN_MLP_Model, MLP_Model, Edge_Embedding_Model, Interaction_Net_Model
from sdk.trained_graph import TrainedGraph
from sdk.pcie_manager import PCIe_Manager
from sdk.ample_driver import Ample_Driver

from tb.variant import Variant
from .utilities import dump_byte_list, binary_to_hex
import argparse


from torch_geometric.data import Data
from collections import defaultdict, deque



 

class Ample():
    def __init__(self, name="ample",
        index=None,
        node_slots = 32,
        message_channel_count = 16,
        precision_count = 1,
        aggregation_buffer_slots = 4
    ):
        self.name = name
        self.model_trace = None
        self.model = None
        self.variant = Variant()
        self.model_map = {
            'GCN_Model': GCN_Model, #TODO
            'GAT_Model': GAT_Model,
            'GIN_Model': GIN_Model,
            'GraphSAGE_Model': GraphSAGE_Model,
            'GCN_MLP_Model': GCN_MLP_Model,
            'MLP_Model': MLP_Model,
            'Edge_Embedding_Model': Edge_Embedding_Model,
            'InteractionNet': Interaction_Net_Model,
            'Interaction_Net_Model': Interaction_Net_Model
        }
        # self.ample = self.connect_to_device()
        self.add_to_device_method()
        self.mem_append= False
        self.driver = Ample_Driver(self.variant)

        self.layer_config_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/layer_config.json"
        self.nodeslot_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/nodeslot_programming.json"
        self.base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config/"
        self.nodeslot_programming_group_start_address = []
        self.nodeslot_mem_dump_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/nodeslots.mem"
        self.sim = True
        self.sim_model_loc = os.environ.get("WORKAREA") + "/hw/sim/layer_config/"

    def add_to_device_method(self):
        ample_instance = self 
        def extended_to_device(model, device, data=None): #Figure out if there is a way to change this
            ample_instance.to_device(model, device,data)
        nn.Module.to_device = extended_to_device


    

    def to_device(self, model, device,data=None):
        self.model = model
        self.inputs = data
        if device == 'ample':
            self.compile(model,data=data,trace_mode='hooks')
            self.overload_forward()
        else:
            print(f'Moving model to {device}')
            model.to(device) 



    def compile(
        self,
        model,
        data,
        base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config",
        precision = 'FLOAT_32',
        reduce =False,
        random = True,
        trained = False,
        plot = True,
        trace_mode = 'hooks'
    ):

        self.model = model
        self.model_name = self.model.__class__.__name__
        self.mem_append= False
        self.nodeslot_programming = []
        if trace_mode == 'fx':
            self.trace_model_fx(self.model, graph_data)
        else:           
            _,input_to_layer_map = self.trace_model_hooks_dataloader_inputs_weights2(self.model, data)
        if plot:
            self.plot_model()

        if self.model_trace is None:
            return

        all_outputs = set()
        for sub_module_name, sub_module_dict in self.model_trace.items():
            all_outputs.update(sub_module_dict['output_names'])

        #Ensure the modules are in the correct order to compile (i.e., input addresses are calculated before module)
        self.model_trace = self.reorder_modules(self.model_trace) 
        
        # Identify external inputs
        external_inputs = set()
        for sub_module_name,sub_module_dict in self.model_trace.items():
            for input_name in sub_module_dict['input_names']:
                if input_name not in all_outputs:
                    external_inputs.add(input_name)

        self.memory_ptr = 0 #keep track of memory address
      

        external_inputs_dict = {element: None for element in external_inputs}
        external_inputs_dict = dict(sorted(external_inputs_dict.items()))
        
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
            assert module_type in self.model_map, f"Module type {module_type} not supported."
           
            edge_attr_messages_addr = None
            in_message_addr = None

            ###DATA###
            #Type 0 - External input - may have external edge input
            if any(name in external_inputs_dict for name in input_names):
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
                        if num_items in edge_num_list:
                            edge_attr_messages_addr = self.model_trace[item_name]['out_addr']
                        else:
                            dataset.num_nodes = self.model_trace[item_name]['num_nodes']
                            sub_module_dict['num_nodes'] = dataset.num_nodes
                            in_message_addr = self.model_trace[item_name]['out_addr']

                        input_nodes.append(item_name)
            
            self.model_trace[sub_module_name]['out_addr'] = self.initialize_sub_model_memory(sub_module_dict['module'],
                                                                                        dataset,
                                                                                        feature_count=32,
                                                                                        sub_model_id = sub_model_id,
                                                                                        in_message_addr=in_message_addr,
                                                                                        edge_attr_messages_addr = edge_attr_messages_addr,
                                                                                        )
        
        self.dump_nodeslot_programming_multi_model()
        self.add_nodeslot_addresses()

        if self.sim:
            self.save_model(self.model,data)
            self.save_graph(data)

    #Initialize memory for node in fx graph sub model
    def initialize_sub_model_memory( #TODO change to submodel
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
         
        #TODO Change to save to intermediate file
        self.init_manager.map_memory(in_messages_addr = in_message_addr,edge_attr_messages_addr=edge_attr_messages_addr) 
        self.init_manager.dump_memory(self.mem_append)
        
        self.nodeslot_programming.append(self.init_manager.return_nodeslot_programming())
        self.memory_ptr,out_messages_address = self.init_manager.dump_layer_config(self.mem_append)

        if self.mem_append == False: #TODO 
            self.mem_append = True
        return out_messages_address


    def trace_model_hooks_dataloader_inputs_weights2(self, model, data):
        self.model_trace = {}
        tensor_id_to_name = {}
        order_counter = 0

        def register_hooks(module, module_name, leaf=False):
            def hook(module, inputs, outputs):
                nonlocal order_counter
                top_level_module_name = module_name.split('.')[0]
                
                # Record the inputs along with their original indices in the forward method
                input_names = []
                input_indices = []
                input_order = []
                for i, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor):
                        tensor_id = id(inp)
                        if tensor_id not in tensor_id_to_name:
                            tensor_name = f"{top_level_module_name}_input_{i}"
                            tensor_id_to_name[tensor_id] = tensor_name
                        else:
                            tensor_name = tensor_id_to_name[tensor_id]
                        input_names.append(tensor_name)
                        input_indices.append(i)  
                        input_order.append(order_counter)
                        order_counter += 1

                output_names = []
                output_order = []
                if isinstance(outputs, (tuple, list)):
                    for i, out in enumerate(outputs):
                        if isinstance(out, torch.Tensor):
                            tensor_id = id(out)
                            tensor_name = f"{top_level_module_name}_output_{i}"
                            tensor_id_to_name[tensor_id] = tensor_name
                            output_names.append(tensor_name)
                            output_order.append(order_counter)
                            order_counter += 1
                else:
                    if isinstance(outputs, torch.Tensor):
                        tensor_id = id(outputs)
                        tensor_name = f"{top_level_module_name}_output_0"
                        tensor_id_to_name[tensor_id] = tensor_name
                        output_names.append(tensor_name)
                        output_order.append(order_counter)
                        order_counter += 1

                weights = None
                if hasattr(module, 'weight') and module.weight is not None:
                    weights = module.weight.data
                elif hasattr(module, 'layers') and len(module.layers) > 0:
                    linear_layer = module.layers[0]
                    if hasattr(linear_layer, 'weight'):
                        weights = linear_layer.weight.data

                module_type = type(module).__name__
                self.model_trace[top_level_module_name] = {
                    'input_names': input_names,
                    'input_indices': input_indices,  
                    'output_names': output_names,
                    'input_order': input_order,
                    'output_order': output_order,
                    'module_type': module_type,
                    'module': module,
                    'weights': weights,  
                    'num_nodes': None,
                    'out_addr': None
                }

            module.register_forward_hook(hook)

            if isinstance(module, torch.nn.Sequential) or module.__module__.startswith("torch_geometric.nn.sequential"):
                for sub_name, sub_module in module.named_children():
                    full_name = f"{module_name}.{sub_name}"
                    register_hooks(sub_module, full_name, leaf=True)

        register_hooks(model, 'model')  
        for name, module in model.named_children():

            register_hooks(module, name)

        model.eval()        
        with torch.no_grad():
            model.forward(*data) 
        
        input_to_layer_map = {}
        for layer_name, trace_info in self.model_trace.items():
            for input_name, input_index in zip(trace_info['input_names'], trace_info['input_indices']):
                input_to_layer_map[(input_name, input_index)] = layer_name
        
        return data, input_to_layer_map

    


    def simulate(self,cpu = False, gpu = False):
        # self.init_manager.save_model()
        # self.init_manager.save_graph()

        args = argparse.Namespace(
            cpu=cpu,
            gpu=gpu,
            sim = True,
            fpga_clk_freq = 250e6,
            device = 'ample',
            preload = False,
            tb_tolerance = 0.01,
            tb_log_level = 'DEBUG',
            build = False,
            gui = False,
            metrics = False,
        )

        bman = BenchmarkingManager(inputs=self.inputs, model=self.model, args=args)
        # if cpu:
        #     metrics = bman.benchmark_cpu()
        # if gpu:
        #     metrics = bman.benchmark_gpu()

        metrics = bman.benchmark()
        #TODO use bman results table
        rows = []
        for component, values in metrics.items():
            for metric, value in values.items():
                formatted_metric = metric.replace("_", " ").replace("-", " ").title()
                formatted_value = f"{value:.6f}" if isinstance(value, float) else f"{value:.6f}"
                rows.append([component, formatted_metric, formatted_value])

        # Create a DataFrame and print it
        df = pd.DataFrame(rows, columns=["Component", "Metric", "Value"])
        print(df.to_markdown(index=False))


    def retrieve_data(self):
        print("Retrieving data from Ample...")
        # Logic to retrieve data from FPGA.


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

    #TODO change to use subdicts
    def trace_model_fx(self, model, dataloader):
        # Use the custom tracer to selectively trace the model
        tracer = CustomTracer(self.model_map)
        traced_graph = tracer.trace(model)

        self.model_trace = {}
        order_counter = 0
        tensor_id_to_name = {}

        for node in traced_graph.nodes:
            if node.op == 'call_module':
                module_type = type(model.get_submodule(node.target)).__name__

                input_names = []
                output_names = []
                for i, inp in enumerate(node.args):
                    tensor_id = id(inp)
                    if tensor_id in tensor_id_to_name:
                        tensor_name = tensor_id_to_name[tensor_id]
                    else:
                        tensor_name = f"{node.name}_input_{i}"
                        tensor_id_to_name[tensor_id] = tensor_name
                    input_names.append(tensor_name)

                for i, out in enumerate(node.users.keys()):
                    tensor_id = id(out)
                    tensor_name = f"{node.name}_output_{i}"
                    tensor_id_to_name[tensor_id] = tensor_name
                    output_names.append(tensor_name)

                self.model_trace[node.name] = (input_names, output_names, order_counter, module_type)
                order_counter += 1

            elif node.op == 'placeholder':
                tensor_id = id(node)
                tensor_name = f"{node.name}_input"
                tensor_id_to_name[tensor_id] = tensor_name
                self.model_trace[node.name] = ([tensor_name], [], order_counter, 'Input')
                order_counter += 1

            elif node.op == 'output':
                input_names = []
                for i, inp in enumerate(node.args):
                    tensor_id = id(inp)
                    if tensor_id in tensor_id_to_name:
                        tensor_name = tensor_id_to_name[tensor_id]
                    else:
                        tensor_name = f"{node.name}_output_{i}"
                        tensor_id_to_name[tensor_id] = tensor_name
                    input_names.append(tensor_name)
                self.model_trace[node.name] = (input_names, [], order_counter, 'Output')
                order_counter += 1

   
  
    def get_node_color(self, module_type):

        # Single module logic
        if module_type == 'Linear':
            return 'orange'
        if module_type == 'Sequential':
            return 'lightblue'
        if module_type == 'LayerNorm':
            return 'pink'
        if module_type == 'ExpandToBatch':
            return 'yellow'
        elif module_type == 'InteractionNet' or module_type == 'Interaction_Net_Model':
            return 'lightgreen'

        return 'white'  # Default color

    
    def plot_model(self, format='png', dpi=100, width=4, height=4):
        dot = Digraph(comment='Simplified Model I/O Graph with Order')
        
        # Set rank direction and size
        dot.attr(rankdir='TB', size=f"{width},{height}!")  # TB for top-bottom layout
        tensor_seen = set()
        
        for sub_module_name, sub_module_dict in self.model_trace.items():
            
            input_names = sub_module_dict['input_names']
            output_names = sub_module_dict['output_names']
            input_order = sub_module_dict['input_order']
            output_order = sub_module_dict['output_order']
            module_type = sub_module_dict['module_type']

            node_color = self.get_node_color(module_type)
            shape = 'ellipse' if node_color != 'white' else 'box'

            annotation = f"{sub_module_name}\nType: {module_type}"
            
            if sub_module_name not in dot.node_attr:
                dot.node(sub_module_name, annotation, shape=shape, style='filled', fillcolor=node_color)

            for i, input_name in enumerate(input_names):
                if input_name not in tensor_seen:
                    tensor_seen.add(input_name)
                    dot.node(input_name, input_name, shape='ellipse')
                dot.edge(input_name, sub_module_name, label=str(input_order))

            for i, output_name in enumerate(output_names):
                if output_name not in tensor_seen:
                    tensor_seen.add(output_name)
                    dot.node(output_name, output_name, shape='ellipse')
                dot.edge(sub_module_name, output_name, label=str(output_order))

        dot.attr(dpi=str(dpi))
               # Specify the output format (e.g., 'png', 'pdf', 'svg', etc.)
        output_format = 'png'  # or any other format you want
        workarea = os.getenv('WORKAREA')

        # Save the file to disk
        print('Rendering graph...')
        output_file = os.path.join(workarea, 'graph_output')  # Name of the output file without extension
        
        dot.render(output_file, format=output_format)


        display(Image(dot.pipe(format=format)))

    def trace_model_hooks_detailed(self, model, dataloader):
        self.model_trace = {}
        tensor_id_to_name = {}
        order_counter = 0

        def register_hooks(module, module_name, leaf=False):
            def hook(module, inputs, outputs):
                nonlocal order_counter
                # Capture the full module name
                full_module_name = module_name
                # Record the inputs
                input_names = []
                input_order = []
                for i, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor):
                        tensor_id = id(inp)
                        if tensor_id not in tensor_id_to_name:
                            tensor_name = f"{full_module_name}_input_{i}"
                            tensor_id_to_name[tensor_id] = tensor_name
                        else:
                            tensor_name = tensor_id_to_name[tensor_id]
                        input_names.append(tensor_name)
                        input_order.append(order_counter)
                        order_counter += 1

                # Record the outputs
                output_names = []
                output_order = []
                if isinstance(outputs, (tuple, list)):
                    for i, out in enumerate(outputs):
                        if isinstance(out, torch.Tensor):
                            tensor_id = id(out)
                            tensor_name = f"{full_module_name}_output_{i}"
                            tensor_id_to_name[tensor_id] = tensor_name
                            output_names.append(tensor_name)
                            output_order.append(order_counter)
                            order_counter += 1
                else:
                    if isinstance(outputs, torch.Tensor):
                        tensor_id = id(outputs)
                        tensor_name = f"{full_module_name}_output_0"
                        tensor_id_to_name[tensor_id] = tensor_name
                        output_names.append(tensor_name)
                        output_order.append(order_counter)
                        order_counter += 1

                # Store the mapping for this module
                module_type = type(module).__name__
                self.model_trace[full_module_name] = (
                    input_names,
                    output_names,
                    input_order + output_order,
                    module_type,
                )

            module.register_forward_hook(hook)

            # If the module is Sequential or any custom sequential container, go one level deeper
            if isinstance(module, torch.nn.Sequential) or module.__module__.startswith(
                "torch_geometric.nn.sequential"
            ):
                for sub_name, sub_module in module.named_children():
                    full_name = f"{module_name}.{sub_name}"
                    register_hooks(sub_module, full_name, leaf=True)

        for name, module in model.named_children():
            register_hooks(module, name)

        # Perform a forward pass using the dataloader to trigger the hooks
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                model.common_step(batch)  # Trigger forward pass
                break  # Only need one batch to trace


    def overload_forward(self):
        original_forward = self.model.forward

        def ample_forward(*args, **kwargs):
            print("Executing on AMPLE")

            if self.sim:
                self.simulate() 
            else:
                self.driver.load_layer_config()
                self.driver.load_regbanks()
                self.driver.execute()


            return original_forward(*args, **kwargs)

        self.model.forward = ample_forward
    

    def build_dependency_graph(self,data):
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        for module_name, module_data in data.items():
            output_names = set(module_data.get('output_names', []))
            for other_module, other_data in data.items():
                if other_module != module_name:
                    if any(input_name in output_names for input_name in other_data.get('input_names', [])):
                        graph[module_name].append(other_module)
                        in_degree[other_module] += 1

        return graph, in_degree

    def topological_sort(self,graph, in_degree):
        queue = deque([node for node in graph if in_degree[node] == 0])
        sorted_list = []

        while queue:
            node = queue.popleft()
            sorted_list.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return sorted_list

    def reorder_modules(self,data):
        if len(data) == 1: #Case where there is only one module no need to reorder
            return data
        graph, in_degree = self.build_dependency_graph(data)
        sorted_modules = self.topological_sort(graph, in_degree)
        return {module_name: data[module_name] for module_name in sorted_modules}

    
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
                # self.nodeslot_programming_group_start_address.append((node_ptr,0))
                
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
        jit_model = torch.jit.trace(self.model, tuple(inputs))
        torch.jit.save(jit_model, self.sim_model_loc + 'model.pt')

        return jit_model

    
    #Save graph for testbench
    def save_graph(self,inputs):
        input_data =inputs

        torch.save({
            'input_data': input_data
        }, self.sim_model_loc + 'graph.pth')

class CustomTracer(fx.Tracer):
    def __init__(self, model_map):
        super().__init__()
        self.model_map = model_map


    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, tuple(self.model_map.values())):
            return True
        return super().is_leaf_module(m, module_qualified_name)
    
