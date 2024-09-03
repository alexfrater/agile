import torch
from sdk.models.models import MLP_Model,Interaction_Net_Model,GCN_Model


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

