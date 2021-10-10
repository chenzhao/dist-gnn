import os
import os.path
import numpy
import scipy.sparse
import torch
from torch_geometric.data import (InMemoryDataset)

"""
TODO: Need to additionally import the following library to use ogb
"""
from ogb.nodeproppred import DglNodePropPredDataset # Load Node Property Prediction datasets in OGB

"""
TODO: set the OGB_DIR to the default download directory for your ogb datasets
"""
OGB_DIR = "/export/data/zhiyuan/CommunityGCN/OGB"

"""
TODO: The class below is modified from Zhao's original template (the older version) for processing the dataset ogbn-products.
The only function that matters is the process() function.
"""

class OgbnProducts(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # return ['reddit_data.npz', 'reddit_graph.npz']
        return ['edge.csv.gz']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # os.unlink(path)
        pass

    def process(self):
        dataset = DglNodePropPredDataset(name="ogbn-products", root=OGB_DIR)
        
        graph, labels = dataset[0]
        labels = torch.flatten(labels)
        n_nodes = graph.num_nodes()
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"].numpy(), split_idx["valid"].numpy(), split_idx["test"].numpy()
        
        x = graph.ndata['feat'].to(torch.float)
        y = labels.to(torch.long)
        split = numpy.ones(n_nodes, dtype=int)
        # split[train_idx] = 1
        split[val_idx] = 2
        split[test_idx] = 3
        split = torch.from_numpy(split)

        edges = graph.edges()
        edge_index = torch.stack([edges[0].to(torch.long), edges[1].to(torch.long)], dim=0)

        torch.save({"x":x, "y":y, "edge_index":edge_index, "split":split}, self.processed_paths[0])