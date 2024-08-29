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
        self.processor = Interaction_Net_Model()
        # self.processor = InteractionNet(
        #     hidden_dim,
        #     hidden_layers=hidden_layers,
        #     aggr=mesh_aggr,
        # )
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
        self.g2m_gnn = Interaction_Net_Model()
        # self.g2m_gnn = InteractionNet(
        #     hidden_dim,
        #     hidden_layers=hidden_layers,
        #     update_edges=False,
        # )
        self.encoding_grid_mlp = make_mlp(
            [hidden_dim] + mlp_blueprint_end
        )

        # decoder
        self.m2g_gnn = Interaction_Net_Model()
        # self.m2g_gnn = InteractionNet(
        #     hidden_dim,
        #     hidden_layers=hidden_layers,
        #     update_edges=False,
        # )

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
        # batch_size = mesh_rep.shape[0]
        m2m_emb = self.m2m_embedder(m2m_features)  # (M_mesh, d_h)
        # m2m_emb = self.m2m_expander(
        #     m2m_emb, batch_size
        # )  # (B, M_mesh, d_h)
 
        # mesh_rep = torch.tensor(mesh_rep)
        # mesh_rep = torch.tensor(mesh_rep)
        mesh_rep = self.processor(
            mesh_rep,m2m_edge_index, m2m_emb
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

        # g2m_gnn_feature = torch.cat((grid_emb, mesh_emb), dim=0)

        # mesh_rep = self.g2m_gnn(
        #     g2m_gnn_feature, g2m_edge_index,g2m_emb
        # )  
        mesh_rep = self.g2m_gnn(
            grid_emb, g2m_edge_index,g2m_emb,mesh_emb
        )  
        
        
        # (B, num_mesh_nodes, d_h)
        # Also MLP with residual for grid representation
        grid_rep =  self.encoding_grid_mlp( #grid_emb + //TODO add identiy connection
            grid_emb
        )  # (B, num_grid_nodes, d_h)

        # Run processor step
        print('forward m2m_gnn')

        mesh_rep = self.process_step(mesh_rep,m2m_edge_index,m2m_features)

        # Map back from mesh to grid
        # m2g_emb = self.m2g_expander(m2g_emb, batch_size)

        print('forward m2g_gnn')
        # m2g_gnn_input_feature = torch.cat((mesh_rep,grid_rep),dim=0)
        # grid_rep = self.m2g_gnn(
        #     m2g_gnn_input_feature,m2g_edge_index, m2g_emb
        # ) 

        grid_rep = self.m2g_gnn(
            mesh_rep, m2g_edge_index, m2g_emb,grid_rep
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
        # print('agg_mlp')
        # print('edge_embed',edge_embed.shape)
        # print('src_embed',src_embed.shape)
        # print('rx_embed',rx_embed.shape)

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

    def forward(self, x, edge_index,edge_attr,y=None):
        # x = x.to(self.precision)  
        if y is not None:
            x = torch.cat((x,y),dim=0)
        print('x',x.shape)
        print('edge_index',edge_index.shape)
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
        
        updated_edge = self.edge_update(src_embed,edge_embed,rx_embed)
        outputs.append(updated_edge)

        rx_node_embed = self.rx_node_embedder(x) #TODO change to x[v] - more efficient
        outputs.append(rx_node_embed)
        rx_aggregated_edges = self.rx_edge_aggr(edge_index,updated_edge) #TODO change to x[v] - more efficient
        outputs.append(rx_aggregated_edges)

        print('rx_node_embed',rx_node_embed.shape)
        print('rx_aggregated_edges',rx_aggregated_edges.shape)

        # updated_node = self.rx_node_update(rx_node_embed,rx_aggregated_edges,0) #TODO fix this

        # outputs.append(updated_node)
        return rx_node_embed
        # return updated_node



