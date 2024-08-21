import os
import numpy as np
import torch
import torch_geometric as pyg
from torch import nn

from neural_lam.interaction_net import InteractionNet


from neural_lam import constants
class GraphLAM_DataProcessor():
    def __init__(self,dataset,graph,device="cpu"):
        self.hierarchical, self.graph_ldict = self.load_graph(graph,device)
        # for name, attr_value in graph_ldict.items():
        #     # Make BufferLists module members and register tensors as buffers
        #     if isinstance(attr_value, torch.Tensor):
        #         self.register_buffer(name, attr_value, persistent=False)
        #     else:
        #         setattr(self, name, attr_value)

        self.static_data_dict = self.load_static_data(dataset)

    def preprocess_inputs(self, dataloader,n_nodes = 50):
        batch = None
        for data_batch in dataloader:
            batch = data_batch  # Trigger forward pass
            break

        (
            init_states,
            target_states,
            batch_static_features,
            forcing_features,
        ) = batch

        # for static_data_name, static_data_tensor in static_data_dict.items():
        #     self.register_buffer(
        #         static_data_name, static_data_tensor, persistent=False
        #     )
        grid_static_features = self.static_data_dict["grid_static_features"]
        prev_state = init_states[:, 1]
        prev_prev_state = init_states[:, 1]
        forcing = forcing_features[:, 0]

        prev_state = prev_state.squeeze(0)
        prev_prev_state= prev_prev_state.squeeze(0)
        forcing=forcing.squeeze(0)
        batch_static_features = batch_static_features.squeeze(0)

        print("prev_state", prev_state.shape)
        print("prev_prev_state", prev_prev_state.shape)
        print("batch_static_features", batch_static_features.shape)
        print("forcing", forcing.shape)
        print("self.grid_static_features", grid_static_features.shape)


        # Create full grid node features of shape (B, num_grid_nodes, grid_dim)
        grid_features = torch.cat(
            (
                prev_state,
                prev_prev_state,
                batch_static_features,
                forcing,
                grid_static_features
                # self.grid_static_features,
            ),
            dim=-1,
        )
        #Not using batch sizes greater than 1 for ample
        grid_features = grid_features.squeeze(0)


        m2m_features = self.graph_ldict["m2m_features"]
        mesh_static_features = self.graph_ldict["mesh_static_features"]
        m2g_features = self.graph_ldict["m2g_features"]
        g2m_features = self.graph_ldict["g2m_features"]
        grid_features = grid_features

        m2m_edge_index = self.graph_ldict["m2m_edge_index"]
        m2g_edge_index = self.graph_ldict["m2g_edge_index"]
        g2m_edge_index = self.graph_ldict["g2m_edge_index"]


        if n_nodes != 0:
            return {
                'features': {
                    'm2m_features': m2m_features[:n_nodes, :],
                    'mesh_static_features': mesh_static_features[:n_nodes, :],
                    'm2g_features': m2g_features[:n_nodes, :],
                    'g2m_features': g2m_features[:n_nodes, :],
                    'grid_features': grid_features[:n_nodes, :]
                },
                'edges': {
                    'm2m_edge_index': self.remap_and_filter_edges(g2m_edge_index, n_nodes),
                    'm2g_edge_index': self.remap_and_filter_edges(m2m_edge_index, n_nodes),
                    'g2m_edge_index': self.remap_and_filter_edges(m2g_edge_index, n_nodes)
                },
                'shapes': {
                    'm2m_features_shape': m2m_features[:n_nodes, :].shape,
                    'mesh_static_features_shape': mesh_static_features[:n_nodes, :].shape,
                    'm2g_features_shape': m2g_features[:n_nodes, :].shape,
                    'g2m_features_shape': g2m_features[:n_nodes, :].shape,
                    'grid_features_shape': grid_features[:n_nodes, :].shape
                }
            }
        else:
            return {
                'features': {
                    'm2m_features': m2m_features,
                    'mesh_static_features': mesh_static_features,
                    'm2g_features': m2g_features,
                    'g2m_features': g2m_features,
                    'grid_features': grid_features
                },
                'edges': {
                    'm2m_edge_index': m2g_edge_index,
                    'm2g_edge_index': m2m_edge_index,
                    'g2m_edge_index': g2m_edge_index,
                },
                'shapes': {
                    'm2m_features_shape': m2m_features.shape,
                    'mesh_static_features_shape': mesh_static_features.shape,
                    'm2g_features_shape': m2g_features.shape,
                    'g2m_features_shape': g2m_features.shape,
                    'grid_features_shape': grid_features.shape,
                }# Keep the shapes as a separate entry
            }
       
        # if n_nodes!=0:
        #     return [ 
        #         [
        #             {'m2m_features': m2m_features[:n_nodes, :]},
        #             {'mesh_static_features': mesh_static_features[:n_nodes, :]},
        #             {'m2g_features': m2g_features[:n_nodes, :]},
        #             {'g2m_features': g2m_features[:n_nodes, :]},
        #             {'grid_features': grid_features[:n_nodes, :]}
        #         ],
        #         [
        #             {'m2m_edge_index':self.remap_and_filter_edges(g2m_edge_index, n_nodes)},
        #             {'m2g_edge_index':self.remap_and_filter_edges(m2m_edge_index, n_nodes)},
        #             {'g2m_edge_index':self.remap_and_filter_edges(m2g_edge_index, n_nodes)}
        #         ]            

        #         ]

        # else:
        #     return [ #TODO change names to features and edges
        #     [
        #         {'m2m_features': m2m_features},
        #         {'mesh_static_features': mesh_static_features},
        #         {'m2g_features': m2g_features},
        #         {'g2m_features': g2m_features},
        #         {'grid_features': grid_features}
        #     ],
        #     [
        #         {'m2m_edge_index':m2g_edge_index},
        #         {'m2g_edge_index':m2m_edge_index},
        #         {'g2m_edge_index':g2m_edge_index}
        #     ]            

        #     ]

  
    def load_static_data(self,dataset_name, device="cpu"):
        """
        Load static files related to dataset
        """
        static_dir_path = os.path.join("data", dataset_name, "static")
        def loads_file(fn):
            return torch.load(
                os.path.join(static_dir_path, fn), map_location=device
            )

        # Load border mask, 1. if node is part of border, else 0.
        border_mask_np = np.load(os.path.join(static_dir_path, "border_mask.npy"))
        border_mask = (
            torch.tensor(border_mask_np, dtype=torch.float32, device=device)
            .flatten(0, 1)
            .unsqueeze(1)
        )  # (N_grid, 1)

        grid_static_features = loads_file(
            "grid_features.pt"
        )  # (N_grid, d_grid_static)

        # Load step diff stats
        step_diff_mean = loads_file("diff_mean.pt")  # (d_f,)
        step_diff_std = loads_file("diff_std.pt")  # (d_f,)

        # Load parameter std for computing validation errors in original data scale
        data_mean = loads_file("parameter_mean.pt")  # (d_features,)
        data_std = loads_file("parameter_std.pt")  # (d_features,)

        # Load loss weighting vectors
        param_weights = torch.tensor(
            np.load(os.path.join(static_dir_path, "parameter_weights.npy")),
            dtype=torch.float32,
            device=device,
        )  # (d_f,)

        return {
            "border_mask": border_mask,
            "grid_static_features": grid_static_features,
            "step_diff_mean": step_diff_mean,
            "step_diff_std": step_diff_std,
            "data_mean": data_mean,
            "data_std": data_std,
            "param_weights": param_weights,
        }


    def remap_and_filter_edges(self,edge_index, num_nodes):

        edge_index[0] = edge_index[0] - edge_index[0][0] #Src Nodes

        flat_edge_index = edge_index.view(-1)
        unique_nodes = torch.unique(flat_edge_index, sorted=False)[:num_nodes]

        node_mapping = torch.full((flat_edge_index.max() + 1,), -1, dtype=torch.long)
        node_mapping[unique_nodes] = torch.arange(len(unique_nodes))

        remapped_src = node_mapping[edge_index[0]]
        remapped_dst = node_mapping[edge_index[1]]

        valid_mask = (remapped_src >= 0) & (remapped_dst >= 0)
        filtered_edge_index = torch.stack([remapped_src[valid_mask], remapped_dst[valid_mask]], dim=0)

        return filtered_edge_index

    def filter_edges(self,edge_index, num_nodes_to_keep):
        # Create a mask that keeps only edges between nodes within the first 50 nodes
        mask = (edge_index[0, :] < num_nodes_to_keep) & (edge_index[1, :] < num_nodes_to_keep)

        # Filter the edge index tensor using the mask
        edge_index_filtered = edge_index[:, mask]
        
        return edge_index_filtered

        
    def load_graph(self,graph_name, device="cpu"):
        """
        Load all tensors representing the graph
        """
        # Define helper lambda function
        graph_dir_path = os.path.join("graphs", graph_name)

        def loads_file(fn):
            return torch.load(os.path.join(graph_dir_path, fn), map_location=device)

        # Load edges (edge_index)
        m2m_edge_index = BufferList(
            loads_file("m2m_edge_index.pt"), persistent=False
        )  # List of (2, M_m2m[l])
        g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
        m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

        n_levels = len(m2m_edge_index)
        hierarchical = n_levels > 1  # Nor just single level mesh graph

        # Load static edge features
        m2m_features = loads_file("m2m_features.pt")  # List of (M_m2m[l], d_edge_f)
        g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
        m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

        # Normalize by dividing with longest edge (found in m2m)
        longest_edge = max(
            torch.max(level_features[:, 0]) for level_features in m2m_features
        )  # Col. 0 is length
        m2m_features = BufferList(
            [level_features / longest_edge for level_features in m2m_features],
            persistent=False,
        )
        g2m_features = g2m_features / longest_edge
        m2g_features = m2g_features / longest_edge

        # Load static node features
        mesh_static_features = loads_file(
            "mesh_features.pt"
        )  # List of (N_mesh[l], d_mesh_static)

        # Some checks for consistency
        assert (
            len(m2m_features) == n_levels
        ), "Inconsistent number of levels in mesh"
        assert (
            len(mesh_static_features) == n_levels
        ), "Inconsistent number of levels in mesh"

        if hierarchical:
            # Load up and down edges and features
            mesh_up_edge_index = BufferList(
                loads_file("mesh_up_edge_index.pt"), persistent=False
            )  # List of (2, M_up[l])
            mesh_down_edge_index = BufferList(
                loads_file("mesh_down_edge_index.pt"), persistent=False
            )  # List of (2, M_down[l])

            mesh_up_features = loads_file(
                "mesh_up_features.pt"
            )  # List of (M_up[l], d_edge_f)
            mesh_down_features = loads_file(
                "mesh_down_features.pt"
            )  # List of (M_down[l], d_edge_f)

            # Rescale
            mesh_up_features = BufferList(
                [
                    edge_features / longest_edge
                    for edge_features in mesh_up_features
                ],
                persistent=False,
            )
            mesh_down_features = BufferList(
                [
                    edge_features / longest_edge
                    for edge_features in mesh_down_features
                ],
                persistent=False,
            )

            mesh_static_features = BufferList(
                mesh_static_features, persistent=False
            )
        else:
            # Extract single mesh level
            m2m_edge_index = m2m_edge_index[0]
            m2m_features = m2m_features[0]
            mesh_static_features = mesh_static_features[0]

            (
                mesh_up_edge_index,
                mesh_down_edge_index,
                mesh_up_features,
                mesh_down_features,
            ) = ([], [], [], [])

        return hierarchical, {
            "g2m_edge_index": g2m_edge_index,
            "m2g_edge_index": m2g_edge_index,
            "m2m_edge_index": m2m_edge_index,
            "mesh_up_edge_index": mesh_up_edge_index,
            "mesh_down_edge_index": mesh_down_edge_index,
            "g2m_features": g2m_features,
            "m2g_features": m2g_features,
            "m2m_features": m2m_features,
            "mesh_up_features": mesh_up_features,
            "mesh_down_features": mesh_down_features,
            "mesh_static_features": mesh_static_features,
        }







class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


#TODO change edge indices to be runtime variable
class GraphLam_model(nn.Module):
    def __init__(self,feature_shape_dict,edge_index_dict,hidden_dim=32,hidden_layers=1,mesh_aggr ='sum'):
        super().__init__()
        mesh_static_features_shape = feature_shape_dict["mesh_static_features_shape"]
        m2m_features_shape = feature_shape_dict["m2m_features_shape"]
        grid_features_shape = feature_shape_dict["grid_features_shape"]
        g2m_features_shape = feature_shape_dict["g2m_features_shape"]
        m2g_features_shape = feature_shape_dict["m2g_features_shape"]
    
        m2m_edge_index = edge_index_dict["m2m_edge_index"]
        g2m_edge_index = edge_index_dict["g2m_edge_index"]
        m2g_edge_index = edge_index_dict["m2g_edge_index"]

        grid_output_dim = (
                constants.GRID_STATE_DIM
        ) 
        
        mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)

    # grid_dim from data + static + batch_static
        (
            num_grid_nodes,
            grid_static_dim,
        ) = grid_features_shape
        mesh_dim = mesh_static_features_shape[1]
        m2m_edges, m2m_dim = m2m_features_shape

        g2m_edges, g2m_dim = g2m_features_shape
        m2g_edges, m2g_dim = m2g_features_shape

        print(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={g2m_edges}, "
            f"m2g={m2g_edges}"
        )

        # Define sub-models
        # Feature embedders for mesh
        self.mesh_embedder = make_mlp([mesh_dim] + mlp_blueprint_end)
        self.m2m_embedder = make_mlp([m2m_dim] + mlp_blueprint_end)

        # GNNs
        # processor

        # Create a single instance of InteractionNet
        processor_net = InteractionNet(
            m2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            aggr=mesh_aggr,
        )
        self.processor = pyg.nn.Sequential(
            "mesh_rep, edge_rep",
            [(processor_net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")]
        )

         # Specify dimensions of data
        num_mesh_nodes, _ = self.get_num_mesh()
        print(
            f"Loaded graph with {num_grid_nodes + num_mesh_nodes} "
            f"nodes ({num_grid_nodes} grid, {num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static + batch_static
        self.g2m_edges, g2m_dim = g2m_features_shape
        self.m2g_edges, m2g_dim = m2g_features_shape

        # Define sub-models
        # Feature embedders for grid
        mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        self.grid_embedder = make_mlp(
            [grid_dim] + mlp_blueprint_end
        )



        # self.g2m_expander = utils.ExpandToBatch()
        # self.m2g_expander = utils.ExpandToBatch()
        # self.mesh_emb_expander = utils.ExpandToBatch()
        # self.grid_features_expander = utils.ExpandToBatch()



        self.g2m_embedder = make_mlp([g2m_dim] + mlp_blueprint_end)
        self.m2g_embedder = make_mlp([m2g_dim] + mlp_blueprint_end)

        # GNNs
        # encoder
        self.g2m_gnn = InteractionNet(
            g2m_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = make_mlp(
            [64] + mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = InteractionNet(
            m2g_edge_index,
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = make_mlp(
            [64] * (hidden_layers + 1)
            + [grid_output_dim],
            layer_norm=False,
        )  # No layer norm on 

    
    def process_step(self, mesh_rep,m2m_features):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embed m2m here first
        batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(m2m_features)  # (M_mesh, d_h)
        # m2m_emb = self.m2m_expander(
        #     m2m_emb, batch_size
        # )  # (B, M_mesh, d_h)
   
        # mesh_rep= mesh_rep.squeeze(0)

        mesh_rep, _ = self.processor(
            mesh_rep, m2m_emb
        )  # (B, N_mesh, d_h)
        return mesh_rep

    def forward(self,grid_features,g2m_features,m2g_features):
        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)

        g2m_emb = self.g2m_embedder(g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.embedd_mesh_nodes(mesh_static_features)

        # print("grid_emb",grid_emb.shape)
        # print("g2m_emb",g2m_emb.shape)
        # print("m2g_emb",m2g_emb.shape)
        # Map from grid to mesh
        # mesh_emb_expanded = self.mesh_emb_expander(
        #     mesh_emb, batch_size
        # )  # (B, num_mesh_nodes, d_h)
        # g2m_emb_expanded = self.g2m_expander(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb, g2m_emb
        )  # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep =  self.encoding_grid_mlp( #grid_emb + //TODO add identiy connection
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        mesh_rep = self.process_step(mesh_rep,m2m_features)

        # Map back from mesh to grid
        # m2g_emb = self.m2g_expander(m2g_emb, batch_size)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_emb
        )  # (B, num_grid_nodes, d_h)

        # Map to output dimension, only for grid
        net_output = self.output_map(
            grid_rep
        )  # (B, num_grid_nodes, d_grid_out)

        return net_output  

    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0


    

def make_mlp(blueprint, layer_norm=True):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)