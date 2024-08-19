

import sys
import os
sys.path.insert(0,'/home/aw1223/ip/agile/imports/neural-lam')
sys.path.insert(0,'/home/aw1223/ip/agile/imports/neural-lam/neural_lam')

sys.path.insert(0,'/home/aw1223/ip/agile')


import torch


from neural_lam.models.graph_lam import GraphLAM
from neural_lam.weather_dataset import WeatherDataset
from neural_lam.utils import make_mlp
from neural_lam.interaction_net import InteractionNet

from sdk.ample import Ample


import argparse
# Manually create the args Namespace object with the desired parameters
args = argparse.Namespace(
    dataset='meps_example',
    model='graph_lam',
    subset_ds=True,
    seed=42,
    n_workers=1,
    epochs=1,
    batch_size=1,
    load=None,
    restore_opt=0,
    precision=32,
    graph='1level',
    hidden_dim=64,
    hidden_layers=1,
    processor_layers=1,
    mesh_aggr='sum',
    output_std=0,
    ar_steps=1,
    loss='wmse',
    step_length=1,
    lr=0.001,
    val_interval=1,
    split='val',
    n_example_pred=1
)
print('Neural LAM Args', args)
#Change to neural-lam directory
# %cd /home/aw1223/ip/worktree_a/imports/neural-lam

new_directory = "/home/aw1223/ip/worktree_a/imports/neural-lam"
os.chdir(new_directory)

eval_loader = torch.utils.data.DataLoader(
                WeatherDataset(
                    args.dataset,
                    pred_length=1,
                    split=args.split,
                    subsample_step=1,
                    subset=bool(args.subset_ds),
                ),
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )


ample = Ample()
model = GraphLAM(args)


model.to_device('ample',data_loader=eval_loader)