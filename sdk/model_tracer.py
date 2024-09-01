import os
import torch.fx as fx
import torch
from collections import defaultdict, deque
from graphviz import Digraph
from IPython.display import Image, display


from sdk.models.models import GCN_Model, GAT_Model, GraphSAGE_Model, GIN_Model, GCN_MLP_Model, MLP_Model, Edge_Embedding_Model, Interaction_Net_Model

class ModelTracer():
    def __init__(self, model_map = None):
        # self.model_map = model_map
        self.model_trace = {}
    
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
        if model_map:
          self.model_map = model_map

    def trace_model(self, model, data,mode = 'hooks'):
      if mode == 'hooks':
        _,input_to_layer_map = self.trace_model_hooks(model, data)
      elif mode == 'fx':
        _,input_to_layer_map = self.trace_model_fx(model, data)
      else:
        print('Invalid mode')
        return
        
      assert self.model_trace, "Model trace is empty"

      self.model_trace = self.reorder_modules(self.model_trace) 
      return self.model_trace

    def trace_model_hooks(self, model, data):
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

    
    def plot_model(self, format='png', dpi=200, width=4, height=4):
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


    def build_dependency_graph(self,data):
      graph = defaultdict(list)
      in_degree = defaultdict(int)
    #   print(data)
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


class CustomTracer(fx.Tracer):
    def __init__(self, model_map):
        super().__init__()
        self.model_map = model_map


    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, tuple(self.model_map.values())):
            return True
        return super().is_leaf_module(m, module_qualified_name)
    
