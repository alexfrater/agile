




# import torch
# import torch.nn as nn

# class ManualGraphInspector:
#     def __init__(self, model: nn.Module):
#         self.model = model

#     def inspect_layers(self, input_data):
#         # Hook to store intermediate outputs
#         def hook_fn(module, input, output):
#             print(f"Layer: {module.__class__.__name__}")
#             print(f"  Input: {input}")
#             print(f"  Output: {output.shape if isinstance(output, torch.Tensor) else output}")
#             print("")

#         # Register hooks on all layers
#         hooks = []
#         for name, layer in self.model.named_modules():
#             if len(list(layer.children())) == 0:  # Register only on leaf modules
#                 hooks.append(layer.register_forward_hook(hook_fn))
        
#         # Perform a forward pass to trigger hooks
#         with torch.no_grad():
#             self.model(input_data)

#         # Remove hooks
#         for hook in hooks:
#             hook.remove()

# # from torch.fx import Tracer, GraphModule, Node

# # class CustomTracer(Tracer):
# #     def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
# #         # Skip tracing for specific layers
# #         if isinstance(m, (GCNConv, InteractionNet)):
# #             return True
# #         return super().is_leaf_module(m, module_qualified_name)

# # class GraphTracer:
# #     def __init__(self, model: nn.Module):
# #         self.model = model
# #         self.graph_module = self.trace_model(model)

# #     def trace_model(self, model):
# #         tracer = CustomTracer()
# #         graph = tracer.trace(model)
# #         return GraphModule(model, graph)

# #     def get_input_output_layers(self):
# #         # ... (your existing method)
# #         pass

# #     def print_input_output_layers(self):
# #         # ... (your existing method)
# #         pass

# #     def draw(self, file="mase_graph.svg"):
# #         drawer = FxGraphDrawer(self.graph_module, "masegraph")
# #         drawer.get_dot_graph().write_svg(file)


# # # import torch
# # # import torch.nn as nn
# # # from torch.fx import Tracer, GraphModule
# # # from torch.fx.passes.graph_drawer import FxGraphDrawer
# # # from torch_geometric.nn import GCNConv
# # # import pydot


# # # class GraphTracer:
# # #     def __init__(self, model: nn.Module):
# # #         self.model = model
# # #         self.graph_module = self.trace_model(model)

# # #     def trace_model(self, model):
# # #         tracer = Tracer()
# # #         graph = tracer.trace(model)
# # #         scripted_model = torch.jit.script(model)
# # #         return scripted_model
# # #         # return GraphModule(model, graph)

# # #     def get_input_output_layers(self):
# # #         inputs_outputs = []
# # #         for node in self.graph_module.graph.nodes:
# # #             print(node)
# # #             if node.op == 'call_module' or node.op == 'call_function':  # Include call_function for wrapped functions
# # #                 if node.op == 'call_module':
# # #                     layer = dict(self.graph_module.named_modules())[node.target]
# # #                 else:
# # #                     layer = node.target  # For call_function, layer is the function itself
# # #                 inputs = node.args
# # #                 outputs = node
# # #                 inputs_outputs.append((layer, inputs, outputs))
# # #         return inputs_outputs




# # #     def print_input_output_layers(self):
# # #         for layer, inputs, outputs in self.get_input_output_layers():
# # #             print(f"Layer: {layer}")
# # #             print(f"  Inputs: {inputs}")
# # #             print(f"  Outputs: {outputs}")
# # #             print("")

# # #     def draw(self, file="mase_graph.svg"):
# # #         drawer = FxGraphDrawer(self.graph_module, "masegraph")
# # #         drawer.get_dot_graph().write_svg(file)

