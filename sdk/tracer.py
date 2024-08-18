import torch
from graphviz import Digraph
from IPython.display import Image, display
from neural_lam.interaction_net import  InteractionNet



import torch
from graphviz import Digraph
from IPython.display import Image, display
from neural_lam.interaction_net import InteractionNet



class Tracer():
  def __init__(self):
    self.trace = 1

  def trace_pl(self, model, dataloader):
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
    io_dict = {}
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
            io_dict[top_level_module_name] = (input_names, output_names, input_order + output_order, module_type)
        
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

    return io_dict

  def get_node_color(self,module_type):
    """
    Returns the color for the node based on the module type.
    """
    if module_type == 'Linear':
        return 'lightblue'
    elif module_type == 'InteractionNetWrapper' or module_type == 'InteractionNet':
        return 'lightgreen'
    elif module_type == 'Sequential':
        return 'orange'
    # Add other module types as needed
    return 'white'  # Default color

  def plot_model(self,model, io_dict, output_file='simplified_model_io_graph', format='png', dpi=300):
      """
      Plots a simplified version of the model's input/output tensor flow using graphviz,
      with numbered connections indicating the order of operations, and includes color-coding
      and annotations for the feature/channel count.

      Args:
          model (pl.LightningModule): The PyTorch Lightning model.
          io_dict (dict): A dictionary where keys are module names and values are tuples of 
                          (input_tensor_names, output_tensor_names, connection_order, module_type).
          output_file (str): The name of the output file (without extension).
          format (str): The format of the output file (e.g., 'png', 'svg', 'pdf').
          dpi (int): The resolution in dots per inch (DPI) for the output image.
      
      Returns:
          Digraph: The generated graph object.
      """
      dot = Digraph(comment='Simplified Model I/O Graph with Order')
      dot.attr(rankdir='TB', size='10')  # TB for top-bottom layout

      # Track unique tensors and simplify connections
      tensor_seen = set()

      for module_name, (input_names, output_names, order, module_type) in io_dict.items():
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

      return dot




# class Tracer():
#   def __init__(self):
#     self.trace = 1

#   def trace_pl(self, model, dataloader):
#       """
#       Traces the PyTorch Lightning model to get the names of input and output tensors 
#       along with the module hierarchy using a provided DataLoader, including the order
#       of connections. It also tracks intermediate variables to ensure they are consistently
#       represented in the diagram.

#       Args:
#           model (pl.LightningModule): The PyTorch Lightning model to trace.
#           dataloader (torch.utils.data.DataLoader): The DataLoader to supply inputs to the model.

#       Returns:
#           dict: A dictionary where keys are module names and values are tuples of (input_tensor_names, output_tensor_names, connection_order).
#       """
#       io_dict = {}
#       tensor_id_to_name = {}
#       order_counter = 0
      
#       def register_hooks(module, module_name):
#           def hook(module, inputs, outputs):
#               nonlocal order_counter
#               # Capture the top-level module name
#               top_level_module_name = module_name.split('.')[0]
#               # Record the inputs
#               input_names = []
#               input_order = []
#               for i, inp in enumerate(inputs):
#                   if isinstance(inp, torch.Tensor):
#                       tensor_id = id(inp)
#                       if tensor_id not in tensor_id_to_name:
#                           tensor_name = f"{top_level_module_name}_input_{i}"
#                           tensor_id_to_name[tensor_id] = tensor_name
#                       else:
#                           tensor_name = tensor_id_to_name[tensor_id]
#                       input_names.append(tensor_name)
#                       input_order.append(order_counter)
#                       order_counter += 1

#               # Record the outputs
#               output_names = []
#               output_order = []
#               if isinstance(outputs, (tuple, list)):
#                   for i, out in enumerate(outputs):
#                       if isinstance(out, torch.Tensor):
#                           tensor_id = id(out)
#                           tensor_name = f"{top_level_module_name}_output_{i}"
#                           tensor_id_to_name[tensor_id] = tensor_name
#                           output_names.append(tensor_name)
#                           output_order.append(order_counter)
#                           order_counter += 1
#               else:
#                   if isinstance(outputs, torch.Tensor):
#                       tensor_id = id(outputs)
#                       tensor_name = f"{top_level_module_name}_output_0"
#                       tensor_id_to_name[tensor_id] = tensor_name
#                       output_names.append(tensor_name)
#                       output_order.append(order_counter)
#                       order_counter += 1

#               # Store the mapping for this module
#               io_dict[top_level_module_name] = (input_names, output_names, input_order + output_order)
          
#           module.register_forward_hook(hook)

#       # Register hooks to top-level modules only
#       for name, module in model.named_children():
#           register_hooks(module, name)

#       # Perform a forward pass using the dataloader to trigger the hooks
#       model.eval()
#       with torch.no_grad():
#           for batch in dataloader:
#               model.common_step(batch)  # Trigger forward pass
#               break  # Only need one batch to trace
#       # print('io_dict', io_dict)
#       for key, value in io_dict.items():
#           print('key', key)
#           print('value', value)
#       return io_dict

#   def get_node_color(self,module):
#       """
#       Returns the color for the node based on the module type.
#       """
#       if isinstance(module, torch.nn.Linear):
#           return 'lightblue'
#       elif isinstance(module, InteractionNet):
#           return 'lightgreen'
#       elif isinstance(module, torch.nn.Sequential):
#           return 'orange'
#       # Add other module types as needed
#       return 'white'  # Default color

#   def plot_model(self,model, io_dict, output_file='simplified_model_io_graph', format='png', dpi=300):
#       """
#       Plots a simplified version of the model's input/output tensor flow using graphviz,
#       with numbered connections indicating the order of operations, and includes color-coding
#       and annotations for the feature/channel count.

#       Args:
#           model (pl.LightningModule): The PyTorch Lightning model.
#           io_dict (dict): A dictionary where keys are module names and values are tuples of 
#                           (input_tensor_names, output_tensor_names, connection_order).
#           output_file (str): The name of the output file (without extension).
#           format (str): The format of the output file (e.g., 'png', 'svg', 'pdf').
#           dpi (int): The resolution in dots per inch (DPI) for the output image.
      
#       Returns:
#           Digraph: The generated graph object.
#       """
#       dot = Digraph(comment='Simplified Model I/O Graph with Order')
#       dot.attr(rankdir='TB', size='10')  # TB for top-bottom layout

#       # Track unique tensors and simplify connections
#       tensor_seen = set()

#       for module_name, (input_names, output_names, order) in io_dict.items():
#           # Extract the top-level module name only
#           top_level_module_name = module_name.split('.')[0]

#           # Get the module and determine its color and shape
#           module = dict(model.named_modules())[top_level_module_name]
#           node_color = self.get_node_color(module)
#           shape = 'box' if node_color != 'white' else 'ellipse'

#           # Annotate the node with feature/channel count
#           annotation = f"{top_level_module_name}\nFeatures: {module.out_features if hasattr(module, 'out_features') else 'N/A'}"
          
#           # Create a node for the top-level module
#           if top_level_module_name not in dot.node_attr:
#               dot.node(top_level_module_name, annotation, shape=shape, style='filled', fillcolor=node_color)

#           # Only show unique input tensors to avoid clutter
#           for i, input_name in enumerate(input_names):
#               if input_name not in tensor_seen:
#                   tensor_seen.add(input_name)
#                   dot.node(input_name, input_name, shape='ellipse')
#               dot.edge(input_name, top_level_module_name, label=str(order[i]))

#           # Only show unique output tensors to avoid clutter
#           for i, output_name in enumerate(output_names):
#               if output_name not in tensor_seen:
#                   tensor_seen.add(output_name)
#                   dot.node(output_name, output_name, shape='ellipse')
#               dot.edge(top_level_module_name, output_name, label=str(order[len(input_names) + i]))

#       # Set output resolution
#       dot.attr(dpi=str(dpi))

#       # Display the graph directly in the notebook
#       # display(Image(dot.pipe(format=format)))

#       return dot

#   # # Initialize your model with the given arguments
#   # hidden_dim = 32
#   # hidden_layers = 1
#   # mesh_aggr = args.mesh_aggr
#   # lr = args.lr
#   # dataset = args.dataset
#   # output_std = args.output_std
#   # loss = args.loss
#   # step_length = args.step_length
#   # n_example_pred = args.n_example_pred
#   # graph = args.graph
#   # processor_layers = args.processor_layers

#   # model = GraphLAM(hidden_dim, hidden_layers, mesh_aggr, lr, dataset, output_std, loss, step_length, n_example_pred, graph, processor_layers)

#   # # Assuming eval_loader is your DataLoader
#   # # eval_loader = torch.utils.data.DataLoader(...)  # Set up your DataLoader

#   # # Trace the model I/O with order and tracking
#   # model_io = trace_lightning_model_io_with_order_and_track(model, eval_loader)

