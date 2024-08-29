import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from graphviz import Digraph
from IPython.display import Image, display
import pypci
import torch.fx as fx

from sdk.initialization_manager import InitManager
from sdk.benchmarking_manager import BenchmarkingManager
from sdk.models.models import GCN_Model, GAT_Model, GraphSAGE_Model, GIN_Model, GCN_MLP_Model, MLP_Model, Edge_Embedding_Model, Interaction_Net_Model
# from neural_lam.interaction_net import InteractionNet
from torch_geometric.datasets import FakeDataset #TODO remove
from sdk.trained_graph import TrainedGraph
from sdk.pcie_manager import PCIe_Manager
from sdk.ample_driver import Ample_Driver
#TODO remove
from sdk.graphs.random_graph import RandomGraph
from tb.variant import Variant
from .utilities import dump_byte_list, binary_to_hex
import argparse

from functools import wraps
# import pyg 
import torch_geometric.nn as pyg_nn

from torch_geometric.data import Data
from collections import defaultdict, deque



# import torch_geometric as pyg
# from torch_geometric.nn import Sequential as pygSequential

# from torch_geometric.nn import Sequential
# from torch_geometric.nn import Sequential as PyGSequential
import torch_geometric as pyg
 

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
        self.nodeslot_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/nodeslot_programming.txt"

        self.nodeslot_programming_group_start_address = []
        self.nodeslot_mem_dump_file = os.environ.get("WORKAREA") + "/hw/sim/layer_config/nodeslots.mem"
        self.sim = False

    def add_to_device_method(self):
        ample_instance = self  # Capture the current Ample instance

        def extended_to_device(model, device, data=None): #Figure out if there is a way to change this
            ample_instance.to_device(model, device,data)
        nn.Module.to_device = extended_to_device


    

    def to_device(self, model, device,data=None):
        self.model = model
        self.inputs = data
        if device == 'ample':
            print('Moving model to Ample')
            print('Compiling model')
            self.compile(model,data=data,trace_mode='hooks')
            print('Overloading')
            self.overload_forward()
            # self.load_nodeslot_file()
            # self.load_memory_file()

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
        plot = False,
        trace_mode = 'hooks'
    ):

        self.model = model
        self.model_name = self.model.__class__.__name__
        self.mem_append= False
        self.nodeslot_programming = []
        print('model name',self.model_name)
        if trace_mode == 'fx':
            self.trace_model_fx(self.model, graph_data)
        else:
            #TODO get rid of data name
            if self.sim:
                self.save_model(self.model,data)
                self.save_graph(data)
                print('Model and graph saved')
            _,input_to_layer_map = self.trace_model_hooks_dataloader_inputs(self.model, data)
            print('input_to_layer_map',input_to_layer_map)
        if plot:
            self.plot_model()

        if self.model_trace is None:
            print("Model tracing failed. Please ensure the model is traceable.")
            return

        all_outputs = set()
        for sub_module_name, sub_module_dict in self.model_trace.items():
            all_outputs.update(sub_module_dict['output_names'])

        #Ensure the modules are in the correct order to compile (i.e., input addresses calculated before module)
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
        

        for key, value in zip(external_inputs_dict.keys(), data):
            external_inputs_dict[key] = value
        print('External inputs',external_inputs_dict)


        # edge_index_inputs = {}


        for key, value in external_inputs_dict.items():
            print('key',key)

            # model_trace[input_name]['num_nodes'] = dataset.num_nodes
                            # if tensor.shape[0] == num_nodes:
                            #     return 'node feature'
                            # elif tensor.shape[0] == num_edges:
                            #     return 'edge attribute'
        #TODO accept external edge attr
        edge_index_external_inputs_dict = {key: value for key, value in external_inputs_dict.items() if isinstance(value, torch.Tensor) and value.shape[0] == 2}

        edge_num_list = []
        for key, value in edge_index_external_inputs_dict.items():
            print('edge_num',value.shape[1])
            edge_num_list.append(value.shape[1])

        edge_attr_external_inputs_dict = {key: value for key, value in external_inputs_dict.items() if isinstance(value, torch.Tensor) and value.shape[0] in edge_num_list}
        #
        # node_feature_external_inputs_dict = {key: value for key, value in external_inputs_dict.items() if isinstance(value, torch.Tensor) and value.shape[0] != 2 & not(value.shape[0]in edge_num_list) }  #change to num_nopdes
        for key, value in external_inputs_dict.items():
            print('key',key)
            print('value',value)
            if isinstance(value, torch.Tensor):
                print('Tensor')
            if value.shape[0] != 2:
                print('not edge index') 

            if key not in edge_attr_external_inputs_dict:
                print('not in edge attr')


        node_feature_external_inputs_dict = {key: value for key, value in external_inputs_dict.items()
                                             if isinstance(value, torch.Tensor) and value.shape[0] != 2 and key not in edge_attr_external_inputs_dict
                                             }

        for key, value in edge_index_external_inputs_dict.items():
            print('key',key)
        for key, value in node_feature_external_inputs_dict.items():
            print('key',key)
        # print('edge_index_inputs',edge_index_inputs)
        # print('node_feature_inputs',node_feature_inputs)
        # print('external_inputs',external_inputs)
        # for key,val in external_inputs: 
        #     print('val',external_input)
        #     print('type',type(val))
        #     if len(val.shape) == 2:
        #     # Check if it's an edge index (typically shape (2, num_edges))
        #         if val.shape[0] == 2:
        #             print('edge index')
        #             edge_index_inputs =  
        #         else:
        #             print('node features')


        for sub_module_name,sub_module_dict in self.model_trace.items():
            input_names = sub_module_dict['input_names']
            module_type = sub_module_dict['module_type']
            print(module_type)
            # assert module_type in self.model_map, f"Module type {module_type} not supported."
            if module_type == 'Sequential':
                module_type  = 'MLP_Model'
            if module_type in self.model_map:
                print('name', sub_module_name)
       
                ###DATA###
                #Type 0 - External input - may have external edge input
                print('check --------')
                print('external_inputs_dict',external_inputs_dict)
                print('input_names',input_names)
                if any(name in external_inputs_dict for name in input_names):
                    print('sub module name external',sub_module_name)

                    #external input module
                    print('sub_module_name',sub_module_name)
                    # print('external_inputs',external_inputs)
                    print('input_names',input_names)
                    dataset = Data()
                    print('edge_attr_external_inputs_dict',edge_attr_external_inputs_dict)
                    print('edge_index_external_inputs_dict',edge_index_external_inputs_dict)
                    print('node_feature_external_inputs_dict',node_feature_external_inputs_dict)
                    # dataset.edge_attr = None
                    for name in input_names:
                        print('name',name)
                        if name in node_feature_external_inputs_dict:
                            print('node features')
                            # if val.shape[0] == 2:
                            #     print('edge index')
                            #     dataset.edge_index = edge_index
                            #     #Edge attr needs to come from previous layer
                            #     # dataset.edge_attr = True
                            # #Check for edge attr
                            # else:
                            print('node features')
                            x = list(node_feature_external_inputs_dict[name])
                            #Change to acccept to X inputs
                            if dataset.x is None:
                                dataset.x = x
                                dataset.num_nodes = len(dataset.x)
                            else:
                                print('dataset.x shape',dataset.x.shape)
                                dataset.x = [torch.cat((dataset.x, x), dim=1)]
                                dataset.num_nodes = dataset.num_nodes + len(x)
                            in_message_addr = None
                            sub_module_dict['num_nodes'] = dataset.num_nodes
                        elif name in edge_index_external_inputs_dict:
                            print('edge index added')
                            dataset.edge_index = edge_index_external_inputs_dict[name]

                        elif name in edge_attr_external_inputs_dict:
                            print('edge attr added')
                            dataset.edge_attr = edge_attr_external_inputs_dict[name]
                        # elif name in edge_attr:
                           
                        #     for item_name, item_data in self.model_trace.items():
                        #         if (name in item_data['output_names']):
                        #             input_node = item_name
                        #             if 'edge_attr' in input_node:
                        #                 in_edge_index_addr = self.model_trace[input_node]['out_addr']
                        #             else:
                        #                 in_message_addr = self.model_trace[input_node
                #Type 0 - Interal input - may have external edge index

                else:
                    dataset = Data()
                    dataset.x = None
                    print('sub module name internal',sub_module_name)

                    #Edge index assingment - Always external
                    for name in input_names:
                        if name in edge_index_external_inputs_dict:
                            dataset.edge_attr = True #TODO check if edge attr is present - use model name
                            dataset.edge_index = edge_index_external_inputs_dict[name]
                            #Find out number of edges to ddetemrine if subqeeunt tensors are edge or node attributes
                            #May break if have same number of edges and nodes
        
                    #Edge and Node Features assignment
                    input_nodes = []
                    for item_name, item_data in self.model_trace.items():
                        print('item_name',item_name)
                        print('item_data',item_data)
                        if any(input_id in item_data['output_names'] for input_id in input_names):
                            print('#############found')

                            input_nodes.append(item_name)
                            #TEMP TODO fix
                            if 'm2g_embedder' in item_name or 'g2m_embedder' in item_name or 'm2m_embedder' in item_name:
                                print('Edge attribute')
                                
                            else:
                                print('Node attribute')
                                dataset.num_nodes = self.model_trace[item_name]['num_nodes']
                                sub_module_dict['num_nodes'] = dataset.num_nodes
                                in_message_addr = self.model_trace[item_name]['out_addr']
                                #
                                print('#############in message loaded')
                            #if edge attr
                            #if node attr

                            # model_trace[input_name]['num_nodes'] = dataset.num_nodes
                            # if tensor.shape[0] == num_nodes:
                            #     return 'node feature'
                            # elif tensor.shape[0] == num_edges:
                            #     return 'edge attribute'
                    
                    #TODO pass in edge index
                    #TODO only picking first input, change to both  
                    # dataset.num_nodes = self.model_trace[input_nodes[0]]['num_nodes'] 
                    # sub_module_dict['num_nodes'] = dataset.num_nodes #Pass down num nodes to next layer
                    # in_message_addr = self.model_trace[input_nodes[0]]['out_addr']
                    # print('input_nodes',input_nodes)
                    # print(self.model_trace[input_nodes[0]])
                    # print('dataset.num_nodes',dataset.num_nodes)

                base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config/" #+'/'  + self.model_name +'/' + sub_module_name
                if dataset.x is not None:
                    sub_model = self.model_map[module_type](dataset.x[0].shape[0]) #TODO get tensor input/output feature widths in trace
                else:
                    sub_model = self.model_map[module_type]()
                    
                # print(self.model_trace[sub_module_name])
                # print('edge_index',dataset.edge_index)

                self.model_trace[sub_module_name]['out_addr'] = self.initialize_node_memory(sub_model,
                                                                                            dataset,
                                                                                            feature_count=32,
                                                                                            in_message_addr=in_message_addr,
                                                                                            base_path=base_path
                                                                                            )
                # print(self.model_trace[sub_module_name])


            else:
                #TODO change neural lam to remove unspoorted modules which are not relevant to prevent this message from showing incorrectly
                print('Module type not supported')
                print(sub_module_name)
        self.dump_nodeslot_programming_mem()
        self.add_nodeslot_addresses()
        

    def trace_model_hooks_dataloader_inputs(self, model, data):
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
                        input_indices.append(i)  # Store the index of the input
                        input_order.append(order_counter)
                        order_counter += 1

                # Record the outputs
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

                # Store the mapping for this module
                module_type = type(module).__name__
                self.model_trace[top_level_module_name] = {
                    'input_names': input_names,
                    'input_indices': input_indices,  # Add input indices to the trace
                    'output_names': output_names,
                    'input_order': input_order,
                    'output_order': output_order,
                    'module_type': module_type,
                    'num_nodes': None,
                    'out_addr': None
                }

            module.register_forward_hook(hook)

            if isinstance(module, torch.nn.Sequential) or module.__module__.startswith("torch_geometric.nn.sequential"):
                for sub_name, sub_module in module.named_children():
                    full_name = f"{module_name}.{sub_name}"
                    register_hooks(sub_module, full_name, leaf=True)

        for name, module in model.named_children():
            register_hooks(module, name)

        # Perform a forward pass using the dataloader to trigger the hooks
        model.eval()        
        with torch.no_grad():
            model.forward(*data)  # Trigger forward pass
        
        # Additional step to map input tensors to the corresponding model layers
        input_to_layer_map = {}
        for layer_name, trace_info in self.model_trace.items():
            for input_name, input_index in zip(trace_info['input_names'], trace_info['input_indices']):
                input_to_layer_map[(input_name, input_index)] = layer_name
        
        return data, input_to_layer_map





    #Initialize memory for node in fx graph
    def initialize_node_memory(
        self,
        model,
        dataset = None,
        feature_count=32,
        in_message_addr = None,
        # model_name = None,
        base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config/",
        precision = 'FLOAT_32',
        reduce =False,
        random = True,
        trained = False,
        date = True
    ):

        print('base_path',base_path)

        d_type = self.get_dtype(precision)
        self.graph = TrainedGraph(dataset,feature_count)
        self.model = model

        self.init_manager = InitManager(self.graph, self.model,self.memory_ptr, base_path=base_path)
        if dataset.x is not None:
            self.init_manager.trained_graph.load_embeddings()
         
        #init_manager.trained_graph.train_embeddings()

        #TODO Change to save to intermeiate file
        self.init_manager.map_memory(in_message_addr) 
        self.init_manager.dump_memory(self.mem_append)
        self.nodeslot_programming.append(self.init_manager.return_nodeslot_programming())
        self.memory_ptr,out_messages_addr = self.init_manager.dump_layer_config(self.mem_append)

        if self.mem_append == False: #TODO 
            print('Writing memory')
            self.mem_append = True
        return out_messages_addr


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
            tb_log_level = 'INFO',
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

   
    def trace_model_hooks_dataloader(self, model, dataloader):
        self.model_trace = {}
        tensor_id_to_name = {}
        order_counter = 0

        def register_hooks(module, module_name,leaf = False):
            def hook(module, inputs, outputs):
                nonlocal order_counter
                # Capture the top-level module name
                top_level_module_name = module_name.split('.')[0]
                # Record the inputs
                input_names = []
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
                        input_order.append(order_counter)
                        order_counter += 1

                # Record the outputs
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

                # Store the mapping for this module
                module_type = type(module).__name__
                # self.model_trace[top_level_module_name] = (input_names, output_names, input_order + output_order, module_type)
                self.model_trace[top_level_module_name] = {
                    'input_names': input_names,
                    'output_names': output_names,
                    'input_order': input_order,
                    'output_order': output_order,
                    'module_type': module_type,
                    'num_nodes': None,
                    'out_addr': None
                }
            module.register_forward_hook(hook)
            # print(module)
            # print(type(module))
            # print('leaf',leaf)
            # If the module is Sequential, go one level deeper
            if isinstance(module, torch.nn.Sequential) or module.__module__.startswith("torch_geometric.nn.sequential"):
                for sub_name, sub_module in module.named_children():
                    full_name = f"{module_name}.{sub_name}"
                    register_hooks(sub_module, full_name,leaf=True)

          

       
           
        for name, module in model.named_children():
            register_hooks(module, name)

        # Perform a forward pass using the dataloader to trigger the hooks
        model.eval()
        data = None
        with torch.no_grad():
            for batch in dataloader:
                data = batch
                model.common_step(batch)  # Trigger forward pass
                break  # Only need one batch to trace
        return data

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

    
    def plot_model(self, format='png', dpi=500):
        dot = Digraph(comment='Simplified Model I/O Graph with Order')
        dot.attr(rankdir='TB', size='10')  # TB for top-bottom layout

        tensor_seen = set()

        print(self.model_trace)
        
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

   #TODO use if depth is et to 1
    def trace_model_hooks_dataloader_sequential(self, model, dataloader):
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
        graph, in_degree = self.build_dependency_graph(data)
        sorted_modules = self.topological_sort(graph, in_degree)
        return {module_name: data[module_name] for module_name in sorted_modules}

    
    def dump_nodeslot_programming_mem(self,append_mode=False):
        mode = 'a' if append_mode else 'w'

        # self.nodeslot_programming = {'nodeslots':[]}
        # self.program_nodeslots()
        # if append_mode:

        with open(self.nodeslot_file,'w') as file:
            json.dump(self.nodeslot_programming, file, indent=4)

        nodeslot_memory_pointer = 0
        nodeslot_byte_list = []
        for group,nodeslot_group in enumerate(self.nodeslot_programming):

            # print('node group',group)
            # print(nodeslot_group)
            self.nodeslot_programming_group_start_address.append(nodeslot_memory_pointer)
            # Dump nodeslots.mem file
            group_byte_list,nmh_length = self.generate_nodeslots_mem(nodeslot_group)
            # print('nhm',nmh_length)
            nodeslot_byte_list +=group_byte_list
            nodeslot_memory_pointer += nmh_length


        dump_byte_list(nodeslot_byte_list, self.nodeslot_mem_dump_file, append_mode)



    def generate_nodeslots_mem(self,nodeslot_group):
        # print('nodeslot_group',nodeslot_group)
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
            if i < len(self.nodeslot_programming_group_start_address): 
                if 'nodeslot_start_address' in layer:
                    layer['nodeslot_start_address'] += self.nodeslot_programming_group_start_address[i]
                else:
                    layer['nodeslot_start_address'] = self.nodeslot_programming_group_start_address[i]
    


        with open(self.layer_config_file, 'w') as file:
            json.dump(data, file, indent=4)

        print("Updated configuration successfully written back to file.")




      #Save JIT model for testbench
    def save_model(self,model,inputs):
        model.eval()
        jit_model = torch.jit.trace(self.model, *inputs)

        torch.jit.save(jit_model, 'model.pt')

        return jit_model

    
    #Save graph for testbench
    def save_graph(self,inputs):
        input_data =inputs
        # input_data = {
        #     'x': self.trained_graph.dataset.x,  
        #     'edge_index': self.trained_graph.dataset.edge_index,
        #     'edge_attr': self.trained_graph.dataset.edge_attr
        # }

        torch.save({
            'input_data': input_data
        }, 'graph.pth')

class CustomTracer(fx.Tracer):
    def __init__(self, model_map):
        super().__init__()
        self.model_map = model_map


    def is_leaf_module(self, m, module_qualified_name):
            # Check if the module is an instance of a class in self.model_map
        if isinstance(m, tuple(self.model_map.values())):
            return True
        return super().is_leaf_module(m, module_qualified_name)
    
#Class to configure and communicate with Ample - pass configured model and target graph