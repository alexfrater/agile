# print('dataset.x' ,dataset.x)
# print('dataset.edge_index' ,dataset.edge_index)
# print('dataset.edge_attr' ,dataset.edge_attr)

import sys
import torch

sys.path.insert(0,'/home/aw1223/ip/agile')

from sdk.ample import Ample

from torch_geometric.datasets import FakeDataset #TODO remove
from sdk.models.models import MLP_Model,Interaction_Net_Model

from torch_geometric.data import Data



class ToyModel(torch.nn.Module):
    def __init__(self, in_channels=32, out_channels=32, layer_count=1, hidden_dimension=32, precision = torch.float32):
        super().__init__()
        self.precision = precision
        self.layers = torch.nn.ModuleList()


        self.linear1 = MLP_Model(in_channels, out_channels) 
        self.linear1.name  = 'linear_embedder1'
        self.layers.append(self.linear1) #Used to map weights in SDK


        self.linear2 = MLP_Model(in_channels, out_channels) 
        self.linear2.name  = 'linear_embedder2'
        self.layers.append(self.linear2) #Used to map weights in SDK

        # self.int_net = Interaction_Net_Model(in_channels, out_channels, layer_count, hidden_dimension, precision)
        # self.int_net.name  = 'int_net'
        # self.layers.append(self.int_net) #Used to map weights in SDK


        for layer in self.layers:
            layer.to(self.precision)

    def forward(self, x):
        outputs_model = []
        x = x.to(self.precision) 
        print(x) 
        outputs_sub_model1,x = self.linear1(x)
        print('outputs_sub_model lin 1')
        print(outputs_sub_model1)
        #need to add a name to each model output as order of computation is not guaranteed
        # outputs_model = outputs_model + outputs_sub_model #Add instead of append to have layer outputs in single list as tb iterates over each layer
        # print(x)

        outputs_sub_model2,x = self.linear2(x)
        print('outputs_sub_model lin 2')
        print(outputs_sub_model2)
        outputs_model = outputs_sub_model1 + outputs_sub_model2
        # _,x = self.int_net(x, edge_index, edge_attr)

        return outputs_model,x

model = ToyModel(32,32)

dataset = Data()
dataset.x = torch.tensor([[-2.0775e-01,  5.4138e-01, -9.3228e-01, -1.5647e-02, -9.2422e-01,
         -1.4551e+00,  1.1447e+00,  4.8944e-01, -1.2190e+00, -2.1434e+00,
          8.0335e-01, -1.3588e+00, -1.2911e+00,  1.1237e+00,  1.1250e-01,
          3.9626e-01,  5.1411e-01,  9.9543e-01,  7.6991e-02,  1.1795e+00,
         -1.2423e+00, -2.6467e-01, -4.9839e-01, -1.0298e+00, -2.2073e+00,
          1.7572e+00, -5.1693e-01,  1.4884e+00,  1.1717e+00, -1.6047e+00,
         -1.0268e+00,  1.6982e+00],
        [-1.8529e+00,  1.6048e+00, -6.7576e-01,  4.8206e-01, -7.3550e-01,
          2.7671e+00,  6.5742e-01,  1.1275e-01, -8.2426e-01,  9.4348e-01,
         -5.4252e-01,  1.3881e+00, -3.2231e-01,  2.2466e+00,  2.9660e-01,
         -3.3892e-01, -3.5646e-01,  1.7096e+00,  6.5559e-01,  9.3671e-01,
         -2.0564e-01, -2.5100e-01, -6.0347e-02, -9.6708e-01,  2.2658e+00,
          2.2228e-01, -2.8040e+00, -8.2614e-01,  5.2462e-01, -2.1695e+00,
         -1.4330e-02,  8.4461e-01],
        [ 6.7803e+00,  5.0186e+00,  5.3118e+00,  6.6033e+00,  5.3946e+00,
          4.0241e+00,  5.3939e+00,  5.0030e+00,  4.8918e+00,  3.9511e+00,
          6.0363e+00,  4.9824e+00,  5.2248e+00,  6.5227e+00,  5.0742e+00,
          4.1273e+00,  4.4227e+00,  5.2284e+00,  5.5740e+00,  6.3545e+00,
          4.3107e+00,  5.2994e+00,  5.3288e+00,  5.3148e+00,  5.8875e+00,
          5.0108e+00,  5.8182e+00,  4.4662e+00,  5.2387e+00,  6.0119e+00,
          4.5336e+00,  5.5987e+00]])
dataset.edge_index = torch.tensor([[ 0,  0,  0,  1,  1,  1, 2,  2,  2],
        [ 0,  1, 2,  0,  1,  2,  0,  1, 2]])
dataset.edge_attr =  None


inputs = [dataset.x]
out = model(*inputs)
import torch.nn.init as init


print('weights')
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)


# %load_ext autoreload
# %autoreload 2
# %pwd
ample = Ample()
ample.sim = True
#Need weights to be initialized before calling to_device
# model = GraphLam_Model(<parameters>)
model.to_device('ample',data=inputs) #Change 

# out = model(*inputs)


out = model(*inputs)
