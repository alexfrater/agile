import random
from collections import defaultdict
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


class FakeDataset(InMemoryDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects.

    Args:
        num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (float, optional): The average degree per node.
            (default: :obj:`10.0`)
        num_channels (int, optional): The number of node features.
            (default: :obj:`64`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        num_classes (int, optional): The number of classes in the dataset.
            (default: :obj:`10`)
        task (str, optional): Whether to return node-level or graph-level
            labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
            If set to :obj:`"auto"`, will return graph-level labels if
            :obj:`num_graphs > 1`, and node-level labels other-wise.
            (default: :obj:`"auto"`)
        is_undirected (bool, optional): Whether the graphs to generate are
            undirected. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        **kwargs (optional): Additional attributes and their shapes
            *e.g.* :obj:`global_features=5`.
    """
    def __init__(
        self,
        num_graphs: int = 1,
        num_nodes: int = 1000,
        avg_degree: float = 10.0,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = 'auto',
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs: Union[int, Tuple[int, ...]],
    ) -> None:
        super().__init__(None, transform)

        if task == 'auto': 
            task = 'graph' if num_graphs > 1 else 'node'
        assert task in ['node', 'graph']

        self.avg_num_nodes = max(avg_num_nodes, int(avg_degree))
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected
        self.kwargs = kwargs

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def generate_data(self) -> Data:
        num_nodes = self.num_nodes #get_num_nodes(self.avg_num_nodes, self.avg_degree)

        data = Data()

        if self._num_classes > 0 and self.task == 'node':
            data.y = torch.randint(self._num_classes, (num_nodes, ))
        elif self._num_classes > 0 and self.task == 'graph':
            data.y = torch.tensor([random.randint(0, self._num_classes - 1)])

        data.edge_index = get_edge_index(num_nodes, num_nodes, self.avg_degree,
                                         self.is_undirected, remove_loops=True)

        if self.num_channels > 0:
            x = torch.randn(num_nodes, self.num_channels)
            if self._num_classes > 0 and self.task == 'node':
                assert isinstance(data.y, Tensor)
                x = x + data.y.unsqueeze(1)
            elif self._num_classes > 0 and self.task == 'graph':
                assert isinstance(data.y, Tensor)
                x = x + data.y
            data.x = x
        else:
            data.num_nodes = num_nodes

        if self.edge_dim > 1:
            data.edge_attr = torch.rand(data.num_edges, self.edge_dim)
        elif self.edge_dim == 1:
            data.edge_weight = torch.rand(data.num_edges)

        for feature_name, feature_shape in self.kwargs.items():
            setattr(data, feature_name, torch.randn(feature_shape))

        return data




def get_edge_index(
    num_src_nodes: int,
    num_dst_nodes: int,
    avg_degree: float,
    is_undirected: bool = False,
    remove_loops: bool = False,
) -> Tensor:

    num_edges = int(num_src_nodes * avg_degree)
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.int64)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.int64)
    edge_index = torch.stack([row, col], dim=0)

    if remove_loops:
        edge_index, _ = remove_self_loops(edge_index)

    num_nodes = max(num_src_nodes, num_dst_nodes)
    if is_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index
