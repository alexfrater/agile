

import sys
import os
workarea = os.environ.get("WORKAREA")

sys.path.insert(0,workarea)


import torch
from torch_geometric.data import Data

def create_grid_graph(rows, cols, num_channels):
    num_nodes = rows * cols
    edge_index = []
    
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            if j < cols - 1:
                edge_index.append([node, node + 1])
            if i < rows - 1:
                edge_index.append([node, node + cols])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.rand((edge_index.size(1), num_channels), dtype=torch.float)
    
    x = torch.rand((num_nodes, num_channels), dtype=torch.float)
    grid_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return grid_data

def create_mesh_graph(num_nodes, num_channels):
    edge_index = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.rand((edge_index.size(1), num_channels), dtype=torch.float)
    
    x = torch.rand((num_nodes, num_channels), dtype=torch.float)
    mesh_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return mesh_data

def create_g2m_and_m2g_connections(grid_data, mesh_data, num_channels):
    grid_nodes = grid_data.num_nodes
    mesh_nodes = mesh_data.num_nodes
    
    g2m_edge_index = torch.stack([torch.randint(0, grid_nodes, (mesh_nodes,)),
                                  torch.arange(0, mesh_nodes)], dim=0)
    
    m2g_edge_index = torch.stack([torch.arange(0, mesh_nodes),
                                  torch.randint(0, grid_nodes, (mesh_nodes,))], dim=0)
    
    g2m_edge_attr = torch.rand((g2m_edge_index.size(1), num_channels), dtype=torch.float)
    m2g_edge_attr = torch.rand((m2g_edge_index.size(1), num_channels), dtype=torch.float)
    
    return g2m_edge_index, g2m_edge_attr, m2g_edge_index, m2g_edge_attr

def create_custom_graph_dataset(grid_rows, grid_cols, mesh_nodes, num_channels, edge_dim):
    grid_data = create_grid_graph(grid_rows, grid_cols, num_channels)
    mesh_data = create_mesh_graph(mesh_nodes, num_channels)
    
    g2m_edge_index, g2m_edge_attr, m2g_edge_index, m2g_edge_attr = create_g2m_and_m2g_connections(grid_data, mesh_data, edge_dim)
    
    combined_x = torch.cat([grid_data.x, mesh_data.x], dim=0)
    
    mesh_edge_index_shifted = mesh_data.edge_index + grid_data.num_nodes
    
    data = Data(
        x=combined_x,
        g2m_edge_index=g2m_edge_index + torch.tensor([[0], [grid_data.num_nodes]], dtype=torch.long),
        g2m_edge_attr=g2m_edge_attr,
        m2m_edge_index=mesh_edge_index_shifted,
        m2m_edge_attr=mesh_data.edge_attr,
        m2g_edge_index=m2g_edge_index + torch.tensor([[grid_data.num_nodes], [0]], dtype=torch.long),
        m2g_edge_attr=m2g_edge_attr
    )
    
    return data

dataset = create_custom_graph_dataset(
    grid_rows=2,
    grid_cols=2,
    mesh_nodes=4,
    num_channels=32,  # Node feature dimension
    edge_dim=32  # Edge feature dimension
)


import torch

from sdk.ample import Ample

from torch_geometric.datasets import FakeDataset #TODO remove
from sdk.models.models import MLP_Model,Interaction_Net_Model,GCN_Model

from torch_geometric.data import Data


class Graphcast(torch.nn.Module):
    def __init__(self, in_channels=32, out_channels=32, layer_count=1, hidden_dimension=32, precision = torch.float32):
        super().__init__()
        self.precision = precision
        self.layers = torch.nn.ModuleList()
       
        self.grid_mesh_embedder = MLP_Model(in_channels, hidden_dimension) 
        self.grid_mesh_embedder.name  = 'grid_mesh_embedder'
        self.layers.append(self.grid_mesh_embedder) 

        self.g2m_embedder = MLP_Model(in_channels, hidden_dimension) 
        self.g2m_embedder.name  = 'g2m_embedder'
        self.layers.append(self.g2m_embedder) 

        self.g2m_int_net = Interaction_Net_Model()
        self.g2m_int_net.name  = 'g2m_int_net'
        self.layers.append(self.g2m_int_net) 

        self.m2m_embedder = MLP_Model(in_channels, hidden_dimension) 
        self.m2m_embedder.name  = 'm2m_embedder'
        self.layers.append(self.m2m_embedder) 

        self.m2m_int_net = Interaction_Net_Model()
        self.m2m_int_net.name  = 'm2m_int_net'
        self.layers.append(self.m2m_int_net) 

        self.m2g_embedder = MLP_Model(in_channels, hidden_dimension) 
        self.m2g_embedder.name  = 'm2m_embedder'
        self.layers.append(self.m2g_embedder) 
 
        self.m2g_int_net = Interaction_Net_Model()
        self.m2g_int_net.name  = 'm2m_int_net'
        self.layers.append(self.m2g_int_net) 
        
        for layer in self.layers:
            layer.to(self.precision)


        #   g2m_edge_attr,
        #   g2m_edge_index,
        #   grid_mesh_rep,
        #   m2m_edge_attr,
        #   m2m_edge_index]
    def forward(
            self,
            g2m_edge_attr,
            g2m_edge_index,
            grid_mesh_rep,
            m2m_edge_attr,
            m2m_edge_index
            # m2g_edge_attr,
            # m2g_edge_index
            ):
            
        outputs_model = []
        
        outputs_sub_model1,grid_mesh_emb = self.grid_mesh_embedder(grid_mesh_rep)
        
        outputs_sub_model2,g2m_emb = self.g2m_embedder(g2m_edge_attr)

        outputs_sub_model3,grid_mesh_emb = self.g2m_int_net(grid_mesh_emb, g2m_edge_index, g2m_emb)
        
        # outputs_sub_model4,m2m_emb = self.m2m_embedder(m2m_edge_attr)

        outputs_sub_model5,grid_mesh_emb = self.m2m_int_net(grid_mesh_emb, m2m_edge_index,g2m_emb)

        # outputs_sub_model6,m2g_emb = self.m2g_embedder(m2g_edge_attr)

        outputs_sub_model7,grid_mesh_emb = self.m2g_int_net(grid_mesh_emb, g2m_edge_index,g2m_emb)

        outputs_model = outputs_sub_model1 + outputs_sub_model2 + outputs_sub_model3 + outputs_sub_model5 + outputs_sub_model7# + outputs_sub_model4  #+ outputs_sub_model6 #+ outputs_sub_model7
        return outputs_model,grid_mesh_emb

model = Graphcast()


print('grid_mesh_rep',dataset.x)
print('g2m_edge_attr',dataset.g2m_edge_attr)
# print('m2m_edge_attr',dataset.m2m_edge_attr)
# print('m2g_edge_attr',dataset.m2g_edge_attr)
print('g2m_edge_index',dataset.g2m_edge_index)
# print('m2m_edge_index',dataset.m2m_edge_index)
# print('m2g_edge_index',dataset.m2g_edge_index)


print('grid_mesh_rep',dataset.x.shape)
print('g2m_edge_attr',dataset.g2m_edge_attr.shape)
# print('m2m_edge_attr',dataset.m2m_edge_attr.shape)
# print('m2g_edge_attr',dataset.m2g_edge_attr.shape)
print('g2m_edge_index',dataset.g2m_edge_index.shape)
# print('m2m_edge_index',dataset.m2m_edge_index.shape)
# print('m2g_edge_index',dataset.m2g_edge_index.shape)

grid_mesh_rep = dataset.x
g2m_edge_attr = dataset.g2m_edge_attr
m2m_edge_attr = dataset.m2m_edge_attr
m2g_edge_attr = dataset.m2g_edge_attr
g2m_edge_index = dataset.g2m_edge_index
m2m_edge_index = dataset.m2m_edge_index
m2g_edge_index = dataset.m2g_edge_index

# inputs  = [g2m_edge_attr, m2m_edge_attr, m2g_edge_attr, g2m_edge_index, m2m_edge_index, m2g_edge_index, grid_mesh_rep]

inputs = [
          g2m_edge_attr,
          g2m_edge_index,
          grid_mesh_rep,
          m2m_edge_attr,
          m2m_edge_index]


dataset = FakeDataset(
                        num_graphs=1, 
                        avg_num_nodes = 100,
                        avg_degree=3,
                        num_channels=32,
                        edge_dim=32
                    )[0]



#IMPORTANT: The order of the inputs in the forward function must match the order of the external inputs identified by the compiler


inputs = [dataset.edge_attr,
          dataset.edge_index,
          dataset.x,
          (dataset.edge_attr + 0.1),
          dataset.edge_index]
outputs_model, grid_mesh_emb = model(*inputs)


from sdk.ample import Ample
ample = Ample(sim=True,cpu_sim=True)
model.to_device('ample',data=inputs)


out = model(*inputs)