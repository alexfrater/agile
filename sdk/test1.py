

import sys
import torch
sys.path.insert(0,"/home/aw1223/agile")



# Print the arguments
print("Arguments passed:", sys.argv)

# Access individual arguments
if len(sys.argv) > 1:
    num_nodes = int(sys.argv[1])
    print("Number of nodes:", num_nodes)
    num_nbrs = int(sys.argv[2])
    print("Number of neighbors:", num_nbrs)


from sdk.models.graphcast import Graphcast
from sdk.datasets.fake_data import FakeDataset

model = Graphcast()
dataset = FakeDataset(
                        num_graphs=1, 
                        num_nodes = num_nodes,
                        degree=num_nbrs,
                        num_channels=32,
                        edge_dim=32
                    )[0]

                
#IMPORTANT: The order of the inputs in the forward function must match the order of the external inputs identified by the compiler
inputs = [dataset.edge_attr,
          dataset.edge_index,
          dataset.x,
          (dataset.edge_attr + 0.1),
          dataset.edge_index]

print(dataset.edge_attr.shape)
print(dataset.edge_index.shape)
print(dataset.x.shape)
outputs_model, grid_mesh_emb = model(*inputs)


from sdk.ample import Ample
ample = Ample(sim=True, cpu_sim = True, gpu_sim=True,plot =False)
model.to_device('ample',data=inputs)
out = model(*inputs)