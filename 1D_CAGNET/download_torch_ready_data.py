# other parts need torch only, no dgl and pyg

import os
import os.path
import numpy
import scipy.sparse
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url, extract_zip)


class TorchGeometricDataSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)


class Reddit(TorchGeometricDataSet):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    raw_file_names = ['reddit_data.npz', 'reddit_graph.npz']
    def __init__(self):
        root = os.path.join('..', 'torch_ready_data', 'Reddit')
        super().__init__(root)

    @staticmethod
    def load_reddit_npz(raw_dir):
        data = numpy.load(os.path.join(raw_dir, 'reddit_data.npz'))
        x = torch.from_numpy(data['feature']).to(torch.float)
        y = torch.from_numpy(data['label']).to(torch.long)
        split = torch.from_numpy(data['node_types'])

        adj = scipy.sparse.load_npz(os.path.join(raw_dir, 'reddit_graph.npz'))
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, x, y, split

    def process(self):
        edge_index, x, y, split = Reddit.load_reddit_npz(self.raw_dir)
        torch.save({"x":x, "y":y, "edge_index":edge_index, "split":split}, self.processed_paths[0])


class SmallerReddit(TorchGeometricDataSet):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    raw_file_names = ['reddit_data.npz', 'reddit_graph.npz']
    def __init__(self):
        root = os.path.join('..', 'torch_ready_data', 'SmallerReddit')
        super().__init__(root)

    def process(self):
        edge_index, x, y, split = Reddit.load_reddit_npz(self.raw_dir)

        max_node = smaller_size = x.size(0)//20
        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)

        torch.save({"x":x[:smaller_size, :], "y":y[:smaller_size],
                    "edge_index":edge_index[:, smaller_mask], "split":split[:smaller_size]}, self.processed_paths[0])


def main():
    Reddit()
    SmallerReddit()
    # Cora()
    # Amazon()
    pass


if __name__ == '__main__':
    main()