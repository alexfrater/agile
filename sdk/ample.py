import os
import pandas as pd

import torch
import torch.nn as nn

from graphviz import Digraph
from IPython.display import Image, display
import pypci
import torch.fx as fx

from sdk.initialization_manager import InitManager
from sdk.benchmarking_manager import BenchmarkingManager
from sdk.models.models import GCN_Model, GAT_Model, GraphSAGE_Model, GIN_Model, GCN_MLP_Model, MLP_Model, Edge_Embedding_Model, Interaction_Net_Model
from neural_lam.interaction_net import InteractionNet

#TODO remove
from sdk.graphs.random_graph import RandomGraph
from tb.variant import Variant

from functools import wraps
# import pyg 
import torch_geometric.nn as pyg_nn

# import torch_geometric as pyg
# from torch_geometric.nn import Sequential as pygSequential

# from torch_geometric.nn import Sequential
# from torch_geometric.nn import Sequential as PyGSequential
import torch_geometric as pyg
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
        self.model_map = {
            'gcn': GCN_Model,
            'gat': GAT_Model,
            'gin': GIN_Model,
            'sage': GraphSAGE_Model,
            'gcn_mlp': GCN_MLP_Model,
            'Sequential': MLP_Model,
            'edge': Edge_Embedding_Model,
            'InteractionNet': InteractionNet
        }
        self.ample = self.connect_to_device()
        self.variant = Variant(message_channel_count, precision_count, aggregation_buffer_slots)
        self.add_to_device_method()
        


    def add_to_device_method(self):
        ample_instance = self  # Capture the current Ample instance

        def extended_to_device(model, device, data_loader=None): #Figure out if there is a way to change this
            ample_instance.to_device(model, device,data_loader)
        nn.Module.to_device = extended_to_device


    def connect_to_device(self):
        devices  = pypci.lspci()
        for device in devices:
            if device.vendor_id ==  0x10EE:
                print('Xilinx device found:')
                print(device)
                #if x :
                    
                #    return ample_pcie name  
        return None

    def to_device(self, model, device,data_loader,graph= None, ):
        if device == 'ample':
            print('Moving model to Ample')
            print('Compiling model')
            self.compile(model,data_loader=data_loader,trace_mode='hooks')
            # self.model = model
        else:
            print(f'Moving model to {device}...')
            model.to(device) 



    def compile(
        self,
        model,
        graph= None,
        graph_data = None,
        data_loader = None,
        base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config",
        precision = 'FLOAT_32',
        reduce =False,
        random = True,
        trained = False,
        plot = True,
        trace_mode = 'hooks'
    ):
       
        self.model = model
        if trace_mode == 'fx':
            self.trace_model_fx(self.model, graph_data)
        else:

            data = self.trace_model_hooks_dataloader(self.model, data_loader)


        all_outputs = set()
        for module_name, (input_names, output_names, order, module_type) in self.model_trace.items():
            all_outputs.update(output_names)

        
        # Identify external inputs
        external_inputs = set()
        for module_name, (input_names, output_names, order, module_type) in self.model_trace.items():
            for input_name in input_names:
                if input_name not in all_outputs:
                    external_inputs.add(input_name)

        ############################
        #TODO Temporary - Need to find a way to map input files/edges/nodes to model - may require new programming model
        inputs_dict = model.preprocess_inputs(data)
        for input_name, input_data in inputs_dict.items():
            print('-' * 40)
            print(input_name)
            # if 'index ' in input_name:
            #     print('Index:', input_data)
            print(input_data)
            print('Shape:', input_data.shape)
        print('External inputs')
        print(external_inputs)
        ############################
        # inputs_dict['edge_index1'] = data.edge_index
        # edge_index = edge_index - edge_index.min(dim=1, keepdim=True)[0]

        

        for name, (input_names, output_names, order, module_type) in self.model_trace.items():
            # assert module_type in self.model_map, f"Module type {module_type} not supported."
            if module_type in self.model_map:
                # model = self.model_map[module_type]()
                print('name', name)
                print(input_names)
                print(output_names)
                print(order)
                print(module_type)
            else:
                #TODO change neural lam to remove unspoorted modules which are not relevant to prevent this message from showing incorrectly
                print('Module type not supported')

        ############################

        if plot:
            self.plot_model()

        if self.model_trace is None:
            print("Model tracing failed. Please ensure the model is traceable.")
            return


        # for name, (input_names, output_names,order, module_type) in self.model_trace.items():
        #     assert module_type in self.model_map, f"Module type {module_type} not supported."

        #     model = self.model_map[module_type]()

        #     #TODO integrate graph
        #     if module_type == 'Sequential':
        #         edge = False
        #     else:
        #         edge = True

        #     #TODO using random graph as dummy data
        #     #TODO fix this : If model does not use edges, dont set edges to be true - will brrak things 
        #     if graph is None:   
        #         graph = RandomGraph(num_nodes=10, avg_degree=1, num_channels=32, graph_precision="FLOAT_32",edge_dim=32,edges = edge) #TODO add var

        #     self.initialize_node_memory(model,graph)
        #     # ample.sim()



    #Initialize memory for node in fx graph
    def initialize_node_memory(
        self,
        model,
        graph,
        base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config",
        precision = 'FLOAT_32',
        reduce =False,
        random = True,
        trained = False
    ):
        
        d_type = self.get_dtype(precision)
        self.graph = graph

        self.model = model
        self.init_manager = InitManager(self.graph, self.model, base_path=base_path)
        # bman = BenchmarkingManager(graph=graph, model=model, args=args)

        self.init_manager.trained_graph.random_embeddings()
            # init_manager.trained_graph.train_embeddings()

        #TODO Change to save to intermeiate file
        self.init_manager.map_memory() 
        self.init_manager.dump_memory()
        self.init_manager.dump_nodeslot_programming()
        self.init_manager.dump_layer_config()



    def sim(self,cpu = True, gpu = False):
        self.init_manager.save_model()
        self.init_manager.save_graph()
        bman = BenchmarkingManager(graph=self.graph, model=self.model)
        if cpu:
            metrics = bman.benchmark_cpu()
        if gpu:
            metrics = bman.benchmark_gpu()

        metrics = bman.benchmark_fpga()
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
                self.model_trace[top_level_module_name] = (input_names, output_names, input_order + output_order, module_type)

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

        # print(self.model_trace)
        
        for module_name, (input_names, output_names, order, module_type) in self.model_trace.items():
            node_color = self.get_node_color(module_type)
            shape = 'ellipse' if node_color != 'white' else 'box'

            annotation = f"{module_name}\nType: {module_type}"
            
            if module_name not in dot.node_attr:
                dot.node(module_name, annotation, shape=shape, style='filled', fillcolor=node_color)

            for i, input_name in enumerate(input_names):
                if input_name not in tensor_seen:
                    tensor_seen.add(input_name)
                    dot.node(input_name, input_name, shape='ellipse')
                dot.edge(input_name, module_name, label=str(order))

            for i, output_name in enumerate(output_names):
                if output_name not in tensor_seen:
                    tensor_seen.add(output_name)
                    dot.node(output_name, output_name, shape='ellipse')
                dot.edge(module_name, output_name, label=str(order))

        dot.attr(dpi=str(dpi))
               # Specify the output format (e.g., 'png', 'pdf', 'svg', etc.)
        output_format = 'png'  # or any other format you want
        workarea = os.getenv('WORKAREA')

        # Save the file to disk
        print('Rendering graph...')
        output_file = os.path.join(workarea, 'graph_output')  # Name of the output file without extension
        
        dot.render(output_file, format=output_format)


        display(Image(dot.pipe(format=format)))


 
    # def to_device(self):
    #     print('Programming Ample device')
    #     #Need su access
    #     self.overload_forward()

 
        # mm = mmap.mmap(f, 0x100, flags=mmap.MAP_SHARED, prot = mmap.PROT_READ,offset=0x0) 
        # m.seek(0x0)#Set offset to feature programming
        #m.write(b'0x0')...
        # m.seek(0x0)#Set offset to nodeslot programming
        #m.write(b'0x0')...

    def overload_forward(self):
        original_forward = self.model.forward

        def ample_forward(*args, **kwargs):
            print("Executing on Ample (sim)")
            print('Writing config over AXI-L')
            if True:
                self.sim() 
            else:
                print('Executing on Ample')   
                # await self.start_clocks()
                # await self.driver.axil_driver.reset_axi_interface()
                # await self.drive_reset()

            return original_forward(*args, **kwargs)

        self.model.forward = ample_forward

    def execute(self, data):
        print("Executing on Ample")



