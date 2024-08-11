import os
import pandas as pd

import torch
import torch.nn as nn

from graphviz import Digraph
from IPython.display import Image, display




from sdk.initialization_manager import InitManager
from sdk.benchmarking_manager import BenchmarkingManager
from sdk.models.models import GCN_Model, GAT_Model, GraphSAGE_Model, GIN_Model, GCN_MLP_Model, MLP_Model, Edge_Embedding_Model, Interaction_Net_Model

#TODO remove
from sdk.graphs.random_graph import RandomGraph


#Class to configure and communicate with Ample - pass configured model and target graph 

class Ample():
    def __init__(self, name="ample", index=None):
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
            'InteractionNet': Interaction_Net_Model
        }


    def compile(
        self,
        model,
        graph= None,
        eval_loader = None,
        base_path = os.environ.get("WORKAREA") + "/hw/sim/layer_config",
        precision = 'FLOAT_32',
        reduce =False,
        random = True,
        trained = False,
        plot = False
    ):
        """
        Compiles the model and graph to be executed on Ample.

        Args:
            model (torch.nn.Module): The PyTorch model to compile.
            graph (torch_geometric.data.Data): The PyTorch Geometric graph to compile.
            base_path (str): The base path to save the compiled model and graph.
            precision (str): The precision to use for the model (FLOAT_32, FIXED_16, FIXED_8, FIXED_4).
            reduce (bool): If True, reduce the model size by removing unused nodes and edges.
            random (bool): If True, initialize the model with random embeddings.
            trained (bool): If True, initialize the model with trained embeddings.
        """
        self.model = model
        self.trace_model_hooks(self.model, eval_loader)
        if plot:
            self.plot_model()
        if self.model_trace is None:
            print("Model tracing failed. Please ensure the model is traceable.")
            return

        for name, (input_names, output_names,order, module_type) in self.model_trace.items():
            assert module_type in self.model_map, f"Module type {module_type} not supported."

            model = self.model_map[module_type]()

            #TODO integrate graph
            if module_type == 'Sequential':
                edge = False
            else:
                edge = True

            #TODO using random graph as dummy data
            #TODO fix this : If model does not use edges, dont set edges to be true - will brrak things 
            if graph is None:   
                graph = RandomGraph(num_nodes=10, avg_degree=1, num_channels=32, graph_precision="FLOAT_32",edge_dim=32,edges = edge) #TODO add var

            self.initialize_memory(model,graph)
            # ample.sim()


    def initialize_memory(
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


    def copy_data_to_device(self, data, sim=True):
        if not sim:
            print("Copying data to the Ample")

    def execute(self, data):
        print("Executing computation on Ample")

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


    

    def trace_model_hooks(self, model, dataloader):
        """
        Traces the PyTorch Lightning model to get the names of input and output tensors 
        along with the module hierarchy using a provided DataLoader, including the order
        of connections. It also tracks intermediate variables to ensure they are consistently
        represented in the diagram.

        Args:
            model (pl.LightningModule): The PyTorch Lightning model to trace.
            dataloader (torch.utils.data.DataLoader): The DataLoader to supply inputs to the model.

        Returns:
            dict: A dictionary where keys are module names and values are tuples of (input_tensor_names, output_tensor_names, connection_order, module_type).
        """
        self.model_trace = {}
        tensor_id_to_name = {}
        order_counter = 0

        def register_hooks(module, module_name):
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

        # Register hooks to top-level modules only
        for name, module in model.named_children():
            register_hooks(module, name)

        # Perform a forward pass using the dataloader to trigger the hooks
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                model.common_step(batch)  # Trigger forward pass
                break  # Only need one batch to trace


    def get_node_color(self,module_type):
        """
        Returns the color for the node based on the module type.
        """
        if module_type == 'Linear':
            return 'lightblue'
        elif module_type == 'Interaction_Net_Model' or module_type == 'Interaction_Net_Model':
            return 'lightgreen'
        elif module_type == 'Sequential':
            return 'orange'
        # Add other module types as needed
        return 'white'  # Default color


    def plot_model(self, format='png', dpi=300):

      dot = Digraph(comment='Simplified Model I/O Graph with Order')
      dot.attr(rankdir='TB', size='10')  # TB for top-bottom layout

      # Track unique tensors and simplify connections
      tensor_seen = set()

      for module_name, (input_names, output_names, order, module_type) in self.model_trace.items():
          # Get the module and determine its color and shape
          node_color = self.get_node_color(module_type)
          shape = 'box' if node_color != 'white' else 'ellipse'

          # Annotate the node with feature/channel count
          annotation = f"{module_name}\nType: {module_type}"
          
          # Create a node for the top-level module
          if module_name not in dot.node_attr:
              dot.node(module_name, annotation, shape=shape, style='filled', fillcolor=node_color)

          # Only show unique input tensors to avoid clutter
          for i, input_name in enumerate(input_names):
              if input_name not in tensor_seen:
                  tensor_seen.add(input_name)
                  dot.node(input_name, input_name, shape='ellipse')
              dot.edge(input_name, module_name, label=str(order[i]))

          # Only show unique output tensors to avoid clutter
          for i, output_name in enumerate(output_names):
              if output_name not in tensor_seen:
                  tensor_seen.add(output_name)
                  dot.node(output_name, output_name, shape='ellipse')
              dot.edge(module_name, output_name, label=str(order[len(input_names) + i]))

      # Set output resolution
      dot.attr(dpi=str(dpi))

      # Display the graph directly in the notebook
      display(Image(dot.pipe(format=format)))
