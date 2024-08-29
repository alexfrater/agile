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

        # print("prev_state", prev_state.shape)
        # print("prev_prev_state", prev_prev_state.shape)
        # print("batch_static_features", batch_static_features.shape)
        # print("forcing", forcing.shape)
        # print("self.grid_static_features", grid_static_features.shape)


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
        # print('gridfeautes shape',grid_features.shape)

        m2m_features = self.graph_ldict["m2m_features"]
        mesh_static_features = self.graph_ldict["mesh_static_features"]
        m2g_features = self.graph_ldict["m2g_features"]
        g2m_features = self.graph_ldict["g2m_features"]
        grid_features = grid_features

        m2m_edge_index = self.graph_ldict["m2m_edge_index"]
        m2g_edge_index = self.graph_ldict["m2g_edge_index"]
        g2m_edge_index = self.graph_ldict["g2m_edge_index"]
        # print("m2g_edge_index",m2g_edge_index)

        # print('grid_features_shape',grid_features.shape)
        # print('mesh_static_features_shape',mesh_static_features.shape)
        if n_nodes != None:
            print('rempaipng')
            # print('m2m_edge_index',m2m_edge_index)
            m2m_edge_index, m2m_features = self.remap_and_filter_edges_mesh(m2m_edge_index, m2m_features, n_nodes)
            # print('m2m_edge_index',m2m_edge_index)
            print('m2g')
            m2g_edge_index, m2g_features = self.remap_and_filter_edges(m2g_edge_index, m2g_features, n_nodes)
            # print('m2g_edge_index',m2g_edge_index)
            print('g2m')
            g2m_edge_index = m2g_edge_index
            g2m_edge_index[0] = m2g_edge_index[1]
            g2m_edge_index[1] = m2g_edge_index[0]
            g2m_features = g2m_features[0:len(g2m_edge_index[1])]
            # print('g2m_edge_index',g2m_edge_index)
            # print('g2m_features',g2m_features)
            # g2m_edge_index, g2m_features = self.remap_and_filter_edges(g2m_edge_index, g2m_features, n_nodes)
            return {
                'features': {
                    'm2m_features': m2m_features,
                    'mesh_static_features': mesh_static_features[:n_nodes, :],
                    'm2g_features': m2g_features,
                    'g2m_features': g2m_features,
                    'grid_features': grid_features[:n_nodes, :]
                },
                'edges': {
                    'm2m_edge_index': m2m_edge_index,
                    'm2g_edge_index': m2g_edge_index,
                    'g2m_edge_index': g2m_edge_index
                },
                'shapes': {
                    'm2m_features_shape': m2m_features.shape,
                    'mesh_static_features_shape': mesh_static_features[:n_nodes, :].shape,
                    'm2g_features_shape': m2g_features.shape,
                    'g2m_features_shape': g2m_features.shape,
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
    
    

        # print('filtered_edge_features',filtered_edge_features.shape)

        return filtered_edge_index, filtered_edge_features

   

    def remap_and_filter_edges_mesh(self,edge_index,edge_features, num_nodes):
        # print('remap')
        # print('edge_index0',edge_index[0])
        # print('edge_index1',edge_index[1])

        # edge_index[0] = edge_index[0] - edge_index[0][0] #Src Nodes
        # print('1edge_index0',edge_index[0])
        # print('1edge_index1',edge_index[1])
        # Create a mask that keeps only edges where both source and destination nodes are less than the threshold
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)

        # Filter the edge index and edge features using the valid mask
        filtered_edge_index = edge_index[:, valid_mask]
        filtered_edge_features = edge_features[valid_mask]
        # print('filtered_edge_index',edge_index[0])
        # print('filtered_edge_index',edge_index[1])
        return filtered_edge_index, filtered_edge_features


    def remap_and_filter_edges(self, edge_index, edge_features, num_nodes=50):

        max_val = edge_index[0].max()
        edge_index[1] = edge_index[1] - max_val

        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)

        filtered_edge_index = edge_index[:, valid_mask] 
        filtered_edge_index[1] = filtered_edge_index[1] + num_nodes
        filtered_edge_features = edge_features[0:len(filtered_edge_index[1])]

        return filtered_edge_index, filtered_edge_features

    
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

