import os
import sys
import numpy as np
import torch
import torch_geometric as pyg
from torch import nn
import torch_scatter


from neural_lam.interaction_net import InteractionNet
sys.path.insert(0,'/home/aw1223/ip/agile')

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
        print('gridfeautes shape',grid_features.shape)

        m2m_features = self.graph_ldict["m2m_features"]
        mesh_static_features = self.graph_ldict["mesh_static_features"]
        m2g_features = self.graph_ldict["m2g_features"]
        g2m_features = self.graph_ldict["g2m_features"]
        grid_features = grid_features

        m2m_edge_index = self.graph_ldict["m2m_edge_index"]
        m2g_edge_index = self.graph_ldict["m2g_edge_index"]
        g2m_edge_index = self.graph_ldict["g2m_edge_index"]
        # print("m2g_edge_index",m2g_edge_index)


        if n_nodes != None:
            return {
                'features': {
                    'm2m_features': m2m_features[:n_nodes, :],
                    'mesh_static_features': mesh_static_features[:n_nodes, :],
                    'm2g_features': m2g_features[:n_nodes, :],
                    'g2m_features': g2m_features[:n_nodes, :],
                    'grid_features': grid_features[:n_nodes, :]
                },
                'edges': {
                    'm2m_edge_index': self.remap_and_filter_edges(m2m_edge_index, n_nodes),
                    'm2g_edge_index': self.remap_and_filter_edges(m2g_edge_index, n_nodes),
                    'g2m_edge_index': self.remap_and_filter_edges(g2m_edge_index, n_nodes)
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
                    'm2m_edge_index': m2m_edge_index,
                    'm2g_edge_index': m2g_edge_index,
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
class GraphLam_Model(nn.Module):
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

        print("m2m_edge_index",m2m_edge_index)
        print("g2m_edge_index",g2m_edge_index)
        print("m2g_edge_index",m2g_edge_index)

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
        self.processor = InteractionNet(
            hidden_dim,
            hidden_layers=hidden_layers,
            aggr=mesh_aggr,
        )
        # self.processor = pyg.nn.Sequential(
        #     "mesh_rep, edge_rep",
        #     [(processor_net, "mesh_rep, mesh_rep, edge_rep -> mesh_rep, edge_rep")]
        # )


        # self.processor = Interaction_Net_Model(,hidden_dim,hidden_layers=hidden_layers,aggr=mesh_aggr)

         # Specify dimensions of data
        num_mesh_nodes = mesh_static_features_shape[0]
        print(
            f"Loaded graph with {num_grid_nodes + num_mesh_nodes} "
            f"nodes ({num_grid_nodes} grid, {num_mesh_nodes} mesh)"
        )

        # grid_dim from data + static + batch_static
        self.g2m_edges, g2m_dim = g2m_features_shape
        self.m2g_edges, m2g_dim = m2g_features_shape

        grid_dim = (
                    2 * constants.GRID_STATE_DIM
                    + grid_static_dim
                    + constants.GRID_FORCING_DIM
                    + constants.BATCH_STATIC_FEATURE_DIM
                )
        grid_dim = grid_features_shape[1]
        # Define sub-models
        # Feature embedders for grid
        mlp_blueprint_end = [hidden_dim] * (hidden_layers + 1)
        self.grid_embedder = make_mlp(
            [grid_dim] + mlp_blueprint_end
        )

        # self.embed_mesh_nodes = make_mlp([mesh_dim] + mlp_blueprint_end)    
        self.mesh_embedder = make_mlp([mesh_dim] + mlp_blueprint_end)

        # self.g2m_expander = utils.ExpandToBatch()
        # self.m2g_expander = utils.ExpandToBatch()
        # self.mesh_emb_expander = utils.ExpandToBatch()
        # self.grid_features_expander = utils.ExpandToBatch()



        self.g2m_embedder = make_mlp([g2m_dim] + mlp_blueprint_end)
        self.m2g_embedder = make_mlp([m2g_dim] + mlp_blueprint_end)

        # GNNs
        # encoder
        self.g2m_gnn = InteractionNet(
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )
        self.encoding_grid_mlp = make_mlp(
            [hidden_dim] + mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = InteractionNet(
            hidden_dim,
            hidden_layers=hidden_layers,
            update_edges=False,
        )

        # Output mapping (hidden_dim -> output_dim)
        self.output_map = make_mlp(
            [hidden_dim] * (hidden_layers + 1)
            + [grid_output_dim],
            layer_norm=False,
        )  # No layer norm on 

    
    def process_step(self, mesh_rep,m2m_edge_index,m2m_features):
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

        mesh_rep = self.processor(
            mesh_rep,mesh_rep,m2m_edge_index, m2m_emb
        )  # (B, N_mesh, d_h)


        return mesh_rep

    # def forward(self,grid_features,mesh_static_features,g2m_features,m2m_features,m2g_features,g2m_edge_index,m2m_edge_index,m2g_edge_index):
    def forward(self,g2m_features,g2m_edge_index,grid_features,m2g_features,m2g_edge_index,m2m_features,mesh_static_features,m2m_edge_index):
        # Embed all features
        grid_emb = self.grid_embedder(grid_features)  # (B, num_grid_nodes, d_h)

        g2m_emb = self.g2m_embedder(g2m_features)  # (M_g2m, d_h)
        m2g_emb = self.m2g_embedder(m2g_features)  # (M_m2g, d_h)
        mesh_emb = self.mesh_embedder(mesh_static_features)

        # print("grid_emb",grid_emb.shape)
        # print("g2m_emb",g2m_emb.shape)
        # print("m2g_emb",m2g_emb.shape)
        # Map from grid to mesh
        # mesh_emb_expanded = self.mesh_emb_expander(
        #     mesh_emb, batch_size
        # )  # (B, num_mesh_nodes, d_h)
        # g2m_emb_expanded = self.g2m_expander(g2m_emb, batch_size)

        # This also splits representation into grid and mesh
        #Temp TODO
        print('forward g2m_gnn')
        print('grid_emb',grid_emb.shape)
        print('mesh_emb',mesh_emb.shape)
        print('g2m_emb',g2m_emb.shape)

        # g2m_gnn_feature = torch.cat((grid_emb, mesh_emb), dim=0)

        # mesh_rep = self.g2m_gnn(
        #     g2m_gnn_feature, g2m_edge_index,g2m_emb
        # )  
        
        mesh_rep = self.g2m_gnn(
            grid_emb, mesh_emb, g2m_edge_index,g2m_emb
        )  
        
        
        # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep =  self.encoding_grid_mlp( #grid_emb + //TODO add identiy connection
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        print('forward m2m_gnn')

        mesh_rep,_ = self.process_step(mesh_rep,m2m_edge_index,m2m_features)

        # Map back from mesh to grid
        # m2g_emb = self.m2g_expander(m2g_emb, batch_size)

        print('forward m2g_gnn')
        # m2g_gnn_input_feature = torch.cat((mesh_rep,grid_rep),dim=0)
        # grid_rep = self.m2g_gnn(
        #     m2g_gnn_input_feature,m2g_edge_index, m2g_emb
        # ) 


        print('mesh_rep',mesh_rep)
        print('grid_rep',grid_rep)
        grid_rep = self.m2g_gnn(
            mesh_rep, grid_rep, m2g_edge_index, m2g_emb
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




#TEMP whilst cant improt from models

class EdgeGCNLayer(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_mlp = nn.Linear(in_channels, out_channels)

    def forward(self, src_embed, rx_embed, edge_embed):
        # Aggregating the embeddings with learnable weights
        combined = src_embed + rx_embed + edge_embed

        # Update with a non-linearity and another MLP layer
        updated_edge_embed = self.update_mlp(combined)
        
        return updated_edge_embed


torch.fx.wrap('agg_mlp')

def agg_mlp(x1, x2, x3):
    model = nn.Sequential(
    nn.Linear(32, 32), 
    )
    output = model(x1 + x2 + x3)

    return output

# class Edge_Embedding_Model(torch.nn.Module): #NodeRx_Src_Embedding_Model
#     def __init__(self):
#         super().__init__()


class AGG_MLP_Model(nn.Module):
    def __init__(self, in_features=32, out_features=32):
        super().__init__()
        # self.add = add_tensors()
        self.in_features = in_features
        self.out_features = out_features
        self.lin = nn.Linear(in_features, out_features,bias = False)

    def forward(self, edge_embed, src_embed, rx_embed):
        # agg = torch.add(edge_embed, src_embed, rx_embed)
        print('agg_mlp')
        print('edge_embed',edge_embed.shape)
        print('src_embed',src_embed.shape)
        print('rx_embed',rx_embed.shape)

        agg = edge_embed + src_embed + rx_embed
        out = self.lin(agg)
        return out

class Edge_Embedding_Model(torch.nn.Module): #NodeRx_Src_Embedding_Model
    def __init__(self, in_channels=32, out_channels=32, layer_count=1, hidden_dimension=32, precision = torch.float32):
        super().__init__()
        self.precision = precision
        self.layers = nn.ModuleList()

        #########Source Node Edge Embed MLP#########
        self.src_embedder = nn.Linear(in_channels, out_channels, bias=False) 
        self.src_embedder.name  = 'linear_src_embedder'
        self.layers.append(self.src_embedder) #Used to map weights in SDK


        #########Edge Node MLP#########
        #Change to GCN to aggregate itself and last layer edge if not first model
        self.edge_embedder = nn.Linear(in_channels, hidden_dimension, bias=False)
        self.edge_embedder.name = 'linear_edge_embedder'
        self.layers.append(self.edge_embedder)


        #########Receive Node Edge Embed MLP#########
        self.rx_embedder = nn.Linear(in_channels, out_channels, bias=False)
        self.rx_embedder.name = 'linear_rx_embedder'
        self.layers.append(self.rx_embedder)


        #########Edge Node Update#########
        self.edge_update = AGG_MLP_Model(in_channels, hidden_dimension)
        self.edge_update.name = 'gcn_edge_update'
        self.layers.append(self.edge_update)


        for layer in self.layers:
            layer.to(self.precision)

    def forward(self, x, edge_index,edge_attr):
        x = x.to(self.precision)  
        outputs = []

        #TODO change to U and V to match with SDK
        u = edge_index[0]
        v = edge_index[1]
        # print('edege_index')
        # print(u)
        # print(v)
        src_embed = self.src_embedder(x)
        outputs.append(src_embed)

        #Check edge attributes are mapped correctly

        edge_embed = self.edge_embedder(edge_attr)
        outputs.append(edge_embed)
        # print('edge_embed')
        # print(edge_embed)

        rx_embed = self.rx_embedder(x)
        outputs.append(rx_embed)

        src_embed = src_embed[u]
        rx_embed = rx_embed[v]
        
        updated_edge = self.edge_update(src_embed,edge_embed,rx_embed)

        outputs.append(updated_edge)

        return outputs



class AggregateEdges(torch.nn.Module):
    def __init__(self,in_channels=32,out_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels,bias=False)

    def forward(self, edge_index, edge_attr):
        # x: Node feature matrix with shape [num_nodes, num_node_features]
        # edge_index: Graph connectivity (edge indices) with shape [2, num_edges]
        # edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        # print('edge_index')
        # print(edge_index)
        # print('rx')
        rx  = edge_index[1]
        # print(rx)
        output = torch_scatter.scatter_add(edge_attr, rx.unsqueeze(1).expand(-1, edge_attr.size(1)), dim=0)
        # x = torch_scatter.scatter_add(edge_attr, rx)
        output = self.lin(output)
        return output
        # return self.propagate(edge_index, x=x, edge_attr=edge_attr)


# class AggregateEdges(MessagePassing):
#     def __init__(self,in_channels=32,out_channels=32):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.lin = torch.nn.Linear(in_channels, out_channels,bias=False)

#     def forward(self, x, edge_index, edge_attr):
#         # Add self loops to the adjacency matrix if needed
#         # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Start propagating messages
#         return self.propagate(edge_index, x=x, edge_attr=edge_attr)

#     def message(self, edge_attr):
#         # Return edge attributes as the message to be passed
#         return edge_attr

#     def aggregate(self, inputs, index):
#     #     # Summing up all incoming edge attributes for each node
#         return torch_scatter.scatter(inputs, index, dim=0, reduce='sum')

#     def update(self, aggr_out):
#         # Return the aggregated result as the updated node features
#         transformed = self.lin(aggr_out)
#         return transformed


class Interaction_Net_Model(torch.nn.Module): #NodeRx_Src_Embedding_Model
    def __init__(self, in_channels=32, out_channels=32, layer_count=1, hidden_dimension=32, precision = torch.float32):
        super().__init__()
        self.precision = precision
        self.layers = nn.ModuleList()

        #########Source Node Edge Embed MLP#########
        self.src_embedder = nn.Linear(in_channels, out_channels, bias=False)
        self.src_embedder.name  = 'linear_src_embedder'
        self.layers.append(self.src_embedder) #Used to map weights in SDK


        #########Edge Node MLP#########
        #Change to GCN to aggregate itself and last layer edge if not first model
        self.edge_embedder = nn.Linear(in_channels, hidden_dimension, bias=False)
        self.edge_embedder.name = 'linear_edge_embedder'
        self.layers.append(self.edge_embedder)


        #########Receive Node Edge Embed MLP#########
        self.rx_embedder = nn.Linear(in_channels, out_channels, bias=False)
        self.rx_embedder.name = 'linear_rx_embedder'
        self.layers.append(self.rx_embedder)


        #########Edge Node GCN#########
        self.edge_update = AGG_MLP_Model(in_channels, hidden_dimension)
        self.edge_update.name = 'gcn_edge_update'
        self.layers.append(self.edge_update)


        #---------- Node Update --------------

        #########Receive Node Embed #########
        self.rx_node_embedder = nn.Linear(in_channels, out_channels, bias=False)
        self.rx_node_embedder.name = 'linear_rx_node_embedder'
        self.layers.append(self.rx_node_embedder)

        #########Receive Node Aggregate Edges #########
        self.rx_edge_aggr = AggregateEdges(in_channels, out_channels)
        self.rx_edge_aggr.name = 'gcn_rx_edge_aggr'
        self.layers.append(self.rx_edge_aggr)


        #########Receive Node Update #########
        self.rx_node_update = AGG_MLP_Model(in_channels, hidden_dimension)
        self.rx_node_update.name = 'rx_node_update'
        self.layers.append(self.rx_node_update)

        for layer in self.layers:
            layer.to(self.precision)

    def forward(self, x, edge_index,edge_attr):
        x = x.to(self.precision)  
        outputs = []

        #TODO change to U and V to match with SDK
        u = edge_index[0] #Source nodes
        v = edge_index[1] #Receive nodes
    
        src_embed = self.src_embedder(x)
        outputs.append(src_embed)


        edge_embed = self.edge_embedder(edge_attr)
        outputs.append(edge_embed)
  
        rx_embed = self.rx_embedder(x)
        outputs.append(rx_embed)


        src_embed = src_embed[u]
        rx_embed = rx_embed[v]
        print('src_embed',src_embed.shape)
        print('rx_embed',rx_embed.shape)


        updated_edge = self.edge_update(src_embed,edge_embed,rx_embed)
        outputs.append(updated_edge)

        rx_node_embed = self.rx_node_embedder(x[v]) #TODO change to x[v] - more efficient
        outputs.append(rx_node_embed)

        rx_aggregated_edges = self.rx_edge_aggr(edge_index,updated_edge) #TODO change to x[v] - more efficient
        outputs.append(rx_aggregated_edges)

        print('rx_aggregated_edges',rx_aggregated_edges.shape)
        print('rx_node_embed',rx_node_embed.shape)

        updated_node = self.rx_node_update(rx_node_embed,rx_aggregated_edges,0)

        outputs.append(updated_node)

        return outputs


# Third-party
import torch
import torch_geometric as pyg
from torch import nn

# First-party
from neural_lam import utils


class InteractionNet(pyg.nn.MessagePassing):
    """
    Implementation of a generic Interaction Network,
    from Battaglia et al. (2016)
    """

    # pylint: disable=arguments-differ
    # Disable to override args/kwargs from superclass

    def __init__(
        self,
        input_dim,
        update_edges=True,
        hidden_layers=1,
        hidden_dim=None,
        edge_chunk_sizes=None,
        aggr_chunk_sizes=None,
        aggr="sum",
    ):
        """
        Create a new InteractionNet

        edge_index: (2,M), Edges in pyg format
        input_dim: Dimensionality of input representations,
            for both nodes and edges
        update_edges: If new edge representations should be computed
            and returned
        hidden_layers: Number of hidden layers in MLPs
        hidden_dim: Dimensionality of hidden layers, if None then same
            as input_dim
        edge_chunk_sizes: List of chunks sizes to split edge representation
            into and use separate MLPs for (None = no chunking, same MLP)
        aggr_chunk_sizes: List of chunks sizes to split aggregated node
            representation into and use separate MLPs for
            (None = no chunking, same MLP)
        aggr: Message aggregation method (sum/mean)
        """
        assert aggr in ("sum", "mean"), f"Unknown aggregation method: {aggr}"
        super().__init__(aggr=aggr)

        if hidden_dim is None:
            # Default to input dim if not explicitly given
            hidden_dim = input_dim

       

        # Create MLPs
        edge_mlp_recipe = [3 * input_dim] + [hidden_dim] * (hidden_layers + 1)
        aggr_mlp_recipe = [2 * input_dim] + [hidden_dim] * (hidden_layers + 1)

        if edge_chunk_sizes is None:
            self.edge_mlp = utils.make_mlp(edge_mlp_recipe)
        else:
            self.edge_mlp = SplitMLPs(
                [utils.make_mlp(edge_mlp_recipe) for _ in edge_chunk_sizes],
                edge_chunk_sizes,
            )

        if aggr_chunk_sizes is None:
            self.aggr_mlp = utils.make_mlp(aggr_mlp_recipe)
        else:
            self.aggr_mlp = SplitMLPs(
                [utils.make_mlp(aggr_mlp_recipe) for _ in aggr_chunk_sizes],
                aggr_chunk_sizes,
            )

        self.update_edges = update_edges

    def forward(self, send_rep, rec_rep, edge_index, edge_rep):
        """
        Apply interaction network to update the representations of receiver
        nodes, and optionally the edge representations.

        send_rep: (N_send, d_h), vector representations of sender nodes
        rec_rep: (N_rec, d_h), vector representations of receiver nodes
        edge_rep: (M, d_h), vector representations of edges used

        Returns:
        rec_rep: (N_rec, d_h), updated vector representations of receiver nodes
        (optionally) edge_rep: (M, d_h), updated vector representations
            of edges
        """
        # send_rep = send_rep.unsqueeze(0)
        # rec_rep = rec_rep.unsqueeze(0)
        # edge_rep = edge_rep.unsqueeze(0)
        # print("send_rep", send_rep.shape)
        # print("rec_rep", rec_rep.shape)
        # print("edge_rep", edge_rep.shape)
        
        # Always concatenate to [rec_nodes, send_nodes] for propagation,
        # but only aggregate to rec_nodes
         # Make both sender and receiver indices of edge_index start at 0
        edge_index = edge_index - edge_index.min(dim=1, keepdim=True)[0]
        # Store number of receiver nodes according to edge_index
        self.num_rec = edge_index[1].max() + 1
        edge_index[0] = (
            edge_index[0] + self.num_rec
        )  # Make sender indices after rec

        node_reps = torch.cat((rec_rep, send_rep), dim=-2)
        edge_rep_aggr, edge_diff = self.propagate(
            edge_index, x=node_reps, edge_attr=edge_rep
        )
        rec_diff = self.aggr_mlp(torch.cat((rec_rep, edge_rep_aggr), dim=-1))

        # Residual connections
        rec_rep = rec_rep + rec_diff

        if self.update_edges:
            edge_rep = edge_rep + edge_diff
            return rec_rep, edge_rep

        return rec_rep.squeeze(0)

    def message(self, x_j, x_i, edge_attr):
        """
        Compute messages from node j to node i.
        """
        return self.edge_mlp(torch.cat((edge_attr, x_j, x_i), dim=-1))

    # pylint: disable-next=signature-differs
    def aggregate(self, inputs, index, ptr, dim_size):
        """
        Overridden aggregation function to:
        * return both aggregated and original messages,
        * only aggregate to number of receiver nodes.
        """
        aggr = super().aggregate(inputs, index, ptr, self.num_rec)
        return aggr, inputs


class SplitMLPs(nn.Module):
    """
    Module that feeds chunks of input through different MLPs.
    Split up input along dim -2 using given chunk sizes and feeds
    each chunk through separate MLPs.
    """

    def __init__(self, mlps, chunk_sizes):
        super().__init__()
        assert len(mlps) == len(
            chunk_sizes
        ), "Number of MLPs must match the number of chunks"

        self.mlps = nn.ModuleList(mlps)
        self.chunk_sizes = chunk_sizes

    def forward(self, x):
        """
        Chunk up input and feed through MLPs

        x: (..., N, d), where N = sum(chunk_sizes)

        Returns:
        joined_output: (..., N, d), concatenated results from the MLPs
        """
        chunks = torch.split(x, self.chunk_sizes, dim=-2)
        chunk_outputs = [
            mlp(chunk_input) for mlp, chunk_input in zip(self.mlps, chunks)
        ]
        return torch.cat(chunk_outputs, dim=-2)
