# other parts need torch only, no dgl and pyg
from typing import Optional, Callable, List

import os
import os.path
import numpy
import scipy.sparse
import torch
import json
from torch_geometric.data import (InMemoryDataset, Data, download_url, extract_zip)


class TorchGeometricDataSet(InMemoryDataset):
    processed_file_names = 'data.pt'
    def __init__(self, root, transform=None, pre_transform=None):
        # os.makedirs(os.path.dirname(root), exist_ok=True)
        super().__init__(root, transform, pre_transform)
        self.data_dict = torch.load(self.processed_paths[0])


    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

class Yelp(TorchGeometricDataSet):
    raw_file_names = ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']
    def __init__(self):
        root = os.path.join('..', 'torch_ready_data', 'Yelp')
        super().__init__(root)

    def download(self):super().download()

    def process(self):
        f = numpy.load(os.path.join(self.raw_dir, 'adj_full.npz'))
        adj = scipy.sparse.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = numpy.load(os.path.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(os.path.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(os.path.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        torch.save({"x":x, "y":y, "edge_index":edge_index, "split":None, "mask":(train_mask, val_mask, test_mask)}, self.processed_paths[0])



class AmazonProducts(InMemoryDataset):
    adj_full_id = '17qhNA8H1IpbkkR-T2BmPQm8QNW5do-aa'
    feats_id = '10SW8lCvAj-kb6ckkfTOC5y0l8XXdtMxj'
    class_map_id = '1LIl4kimLfftj4-7NmValuWyCQE8AaE7P'
    role_id = '1npK9xlmbnjNkV80hK2Q68wTEVOFjnt4K'
    def __init__(self, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        root = os.path.join('..', 'torch_ready_data', 'AmazonProducts')
        super().__init__(root, transform, pre_transform)
    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    def download(self):
        from google_drive_downloader import GoogleDriveDownloader as gdd
        path = os.path.join(self.raw_dir, 'adj_full.npz')
        gdd.download_file_from_google_drive(self.adj_full_id, path)
        path = os.path.join(self.raw_dir, 'feats.npy')
        gdd.download_file_from_google_drive(self.feats_id, path)
        path = os.path.join(self.raw_dir, 'class_map.json')
        gdd.download_file_from_google_drive(self.class_map_id, path)
        path = os.path.join(self.raw_dir, 'role.json')
        gdd.download_file_from_google_drive(self.role_id, path)

    def process(self):
        f = numpy.load(os.path.join(self.raw_dir, 'adj_full.npz'))
        adj = scipy.sparse.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = numpy.load(os.path.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(os.path.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(os.path.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        torch.save({"x":x, "y":y, "edge_index":edge_index, "split":None, "mask":(train_mask, val_mask, test_mask)}, self.processed_paths[0])


class Flickr(InMemoryDataset):
    adj_full_id = '1crmsTbd1-2sEXsGwa2IKnIB7Zd3TmUsy'
    feats_id = '1join-XdvX3anJU_MLVtick7MgeAQiWIZ'
    class_map_id = '1uxIkbtg5drHTsKt-PAsZZ4_yJmgFmle9'
    role_id = '1htXCtuktuCW8TR8KiKfrFDAxUgekQoV7'
    def __init__(self, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        root = os.path.join('..', 'torch_ready_data', 'Flickr')
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        return ['adj_full.npz', 'feats.npy', 'class_map.json', 'role.json']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        from google_drive_downloader import GoogleDriveDownloader as gdd

        path = os.path.join(self.raw_dir, 'adj_full.npz')
        gdd.download_file_from_google_drive(self.adj_full_id, path)

        path = os.path.join(self.raw_dir, 'feats.npy')
        gdd.download_file_from_google_drive(self.feats_id, path)

        path = os.path.join(self.raw_dir, 'class_map.json')
        gdd.download_file_from_google_drive(self.class_map_id, path)

        path = os.path.join(self.raw_dir, 'role.json')
        gdd.download_file_from_google_drive(self.role_id, path)

    def process(self):
        f = numpy.load(os.path.join(self.raw_dir, 'adj_full.npz'))
        adj = scipy.sparse.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = numpy.load(os.path.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(os.path.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(os.path.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        torch.save({"x":x, "y":y, "edge_index":edge_index, "split":None, "mask":(train_mask, val_mask, test_mask)}, self.processed_paths[0])



class Reddit(TorchGeometricDataSet):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    raw_file_names = ['reddit_data.npz', 'reddit_graph.npz']
    def __init__(self):
        root = os.path.join('..', 'torch_ready_data', 'Reddit')
        super().__init__(root)

    def download(self):super().download()

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

    def download(self):super().download()
    def process(self):
        edge_index, x, y, split = Reddit.load_reddit_npz(self.raw_dir)

        max_node = smaller_size = x.size(0)//20
        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)

        torch.save({"x":x[:smaller_size, :].clone(), "y":y[:smaller_size].clone(),
                    "edge_index":edge_index[:, smaller_mask].clone(), "split":split[:smaller_size].clone()}, self.processed_paths[0])

class OneQuarterReddit(TorchGeometricDataSet):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    raw_file_names = ['reddit_data.npz', 'reddit_graph.npz']
    def __init__(self):
        root = os.path.join('..', 'torch_ready_data', 'OneQuarterReddit')
        super().__init__(root)

    def download(self):super().download()
    def process(self):
        edge_index, x, y, split = Reddit.load_reddit_npz(self.raw_dir)

        max_node = smaller_size = x.size(0)//4
        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)

        torch.save({"x":x[:smaller_size, :].clone(), "y":y[:smaller_size].clone(),
                    "edge_index":edge_index[:, smaller_mask].clone(), "split":split[:smaller_size].clone()}, self.processed_paths[0])

class TinyReddit(TorchGeometricDataSet):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    raw_file_names = ['reddit_data.npz', 'reddit_graph.npz']
    def __init__(self):
        root = os.path.join('..', 'torch_ready_data', 'TinyReddit')
        super().__init__(root)

    def download(self):super().download()
    def process(self):
        edge_index, x, y, split = Reddit.load_reddit_npz(self.raw_dir)

        max_node = smaller_size = 64
        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)

        torch.save({"x":x[:smaller_size, :].clone(), "y":y[:smaller_size].clone(),
                    "edge_index":edge_index[:, smaller_mask].clone(), "split":split[:smaller_size].clone()}, self.processed_paths[0])


def main():
    # Reddit()
    # TinyReddit()
    # SmallerReddit()
    # OneQuarterReddit()
    # Cora()
    # AmazonProducts()
    Flickr()
    # Yelp()
    pass


if __name__ == '__main__':
    main()
