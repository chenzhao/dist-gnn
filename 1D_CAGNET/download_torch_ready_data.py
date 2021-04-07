# other parts need torch only, no dgl and pyg

import os
import os.path
import numpy
import scipy.sparse
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url, extract_zip)



class Reddit(InMemoryDataset):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data = numpy.load(os.path.join(self.raw_dir, 'reddit_data.npz'))
        x = torch.from_numpy(data['feature']).to(torch.float)
        y = torch.from_numpy(data['label']).to(torch.long)
        split = torch.from_numpy(data['node_types'])

        adj = scipy.sparse.load_npz(os.path.join(self.raw_dir, 'reddit_graph.npz'))
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        torch.save({"x":x, "y":y, "edge_index":edge_index, "split":split}, self.processed_paths[0])


class SmallerReddit(InMemoryDataset):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['reddit_data.npz', 'reddit_graph.npz']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data = numpy.load(os.path.join(self.raw_dir, 'reddit_data.npz'))
        x = torch.from_numpy(data['feature']).to(torch.float)
        y = torch.from_numpy(data['label']).to(torch.long)
        split = torch.from_numpy(data['node_types'])

        adj = scipy.sparse.load_npz(os.path.join(self.raw_dir, 'reddit_graph.npz'))
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        # smaller
        max_node = smaller_size = x.size(0)//20

        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)
        smaller_edge_index = edge_index[:,smaller_mask]
        smaller_x = x[:smaller_size, :]
        smaller_y = y[:smaller_size]
        smaller_split = split[:smaller_size]

        torch.save({"x":smaller_x, "y":smaller_y, "edge_index":smaller_edge_index, "split":smaller_split}, self.processed_paths[0])


def main():
    data_root = os.path.join('..', 'torch_ready_data', 'Reddit')
    Reddit(data_root)
    data_root = os.path.join('..', 'torch_ready_data', 'SmallerReddit')
    SmallerReddit(data_root)
    # Cora()
    # Amazon()
    pass


if __name__ == '__main__':
    main()