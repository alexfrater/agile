
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.insert(0,'/home/aw1223/ip/agile')
sys.path.insert(0,'/home/aw1223/ip/agile/imports/neural-lam')
sys.path.insert(0,'/home/aw1223/ip/agile/imports/neural-lam/neural_lam')
sys.path.insert(0,'/home/aw1223/ip/worktree_a/imports/neural-lam')

import torch
from sdk.models.graph_lam import GraphLAM_DataProcessor
from sdk.models.graph_lam import GraphLam_Model
from neural_lam.weather_dataset import WeatherDataset

dataset='meps_example'

eval_loader = torch.utils.data.DataLoader(
                WeatherDataset(
                    dataset,
                    pred_length=1,
                    split='test',
                    subsample_step=1,
                ),
                batch_size=1,
                shuffle=False,
                num_workers=1,
            )

data_processor = GraphLAM_DataProcessor(dataset,graph='1level')

graph_data = data_processor.preprocess_inputs(eval_loader)
# print(graph_data)

feature_shape_dict = graph_data['shapes']
edge_index_dict = graph_data['edges']
# print(edge_index_dict)

model = GraphLam_Model(feature_shape_dict,edge_index_dict)
# #Change data to a list of inputs
# print(graph_data['features'])

# print('grid_features',graph_data['features']['grid_features'].shape)
# print('mesh_static_features',graph_data['features']['mesh_static_features'].shape)
# print('g2m_features',graph_data['features']['g2m_features'].shape)
# print('m2m_features',graph_data['features']['m2m_features'].shape)
# print('m2g_features',graph_data['features']['m2g_features'].shape)


grid_features = graph_data['features']['grid_features']
mesh_static_features = graph_data['features']['mesh_static_features']
g2m_features = graph_data['features']['g2m_features']
m2m_features = graph_data['features']['m2m_features']
m2g_features = graph_data['features']['m2g_features']
print('grid_features',grid_features.shape)
print('mesh_static_features',mesh_static_features.shape)
print('g2m_features',g2m_features.shape)
print('m2m_features',m2m_features.shape)
print('m2g_features',m2g_features.shape)



g2m_edge_index = graph_data['edges']['g2m_edge_index']
m2m_edge_index = graph_data['edges']['m2m_edge_index']
m2g_edge_index = graph_data['edges']['m2g_edge_index']

# External inputs {'g2m_embedder_input_0': None, 'g2m_gnn_input_2': None, 'grid_embedder_input_0': None, 'm2g_embedder_input_0': None, 'm2g_gnn_input_2': None, 'm2m_embedder_input_0': None, 'mesh_embedder_input_0': None, 'processor_input_2': None}
#Must sort inputs
inputs = [g2m_features,g2m_edge_index,grid_features,m2g_features,m2g_edge_index,m2m_features,mesh_static_features,m2m_edge_index]
# features = [grid_features,mesh_static_features,g2m_features,m2m_features,m2g_features]
# edges = [g2m_edge_index,m2m_edge_index,m2g_edge_index]
# inputs = [features,edges]
print('inputs')
print(inputs)
# out = model(grid_features =grid_features ,mesh_static_features=mesh_static_features,g2m_features=g2m_features,m2m_features=m2m_features,m2g_features=m2g_features)
out = model(*inputs)
# out = model(*features,*edges)
print(out)

