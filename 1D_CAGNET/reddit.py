import os
import datetime as dt
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch

import torch_geometric as pyg
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)

from torch_sparse import coalesce
from torch_geometric.utils.convert import to_networkx

print(__file__, 'imported')
#import networkx 
#import metis


class SmallerReddit(InMemoryDataset):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = torch.from_numpy(data['feature']).to(torch.float)
        y = torch.from_numpy(data['label']).to(torch.long)
        split = torch.from_numpy(data['node_types'])


        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        # smaller
        max_node = smaller_size = x.size(0)//10

        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)
        smaller_edge_index = edge_index[:,smaller_mask]
        smaller_x = x[:smaller_size, :]
        smaller_y = y[:smaller_size]
        smaller_split = split[:smaller_size]

        data = Data(x=smaller_x, edge_index=smaller_edge_index, y=smaller_y)
        data.train_mask = smaller_split == 1
        data.val_mask = smaller_split == 2
        data.test_mask = smaller_split == 3

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])



class Reddit(InMemoryDataset):
    url = 'https://data.dgl.ai/dataset/reddit.zip'
    def __init__(self, root, transform=None, pre_transform=None):
        super(Reddit, self).__init__(root, transform, pre_transform)
        # self.process_with_partition()
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.partition(self[0])
        # torch.save(self.collate([self.data]), self.processed_paths[0])
        # print('partitioned graph saved')

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
        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = torch.from_numpy(data['feature']).to(torch.float)
        y = torch.from_numpy(data['label']).to(torch.long)
        split = torch.from_numpy(data['node_types'])

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def partition(self, pyg_graph, P=8):
        g = to_networkx(pyg_graph)
        print('partition begin')
        (edgecuts, parts) = metis.part_graph(g, P)
        print('parted', edgecuts, len(parts))
        from collections import Counter
        c = Counter(parts)
        print(c.most_common())

        node_ids = list(g)
        # print(parts)
        # print(node_ids)
        # print(pyg_graph.edge_index)
        self.swap_node_by_parts(pyg_graph, parts, node_ids)
        print('node swapped')
        # print(pyg_graph.edge_index)
        # print(parts)

    def swap_node_by_parts(self, pyg_graph, parts, node_ids):
        # parts        = 0,1,0,1,0,1
        # node_ids     = 1,2,5,6,7,8
        # will sort to = 1,5,7,2,6,8
        # e.g. id=5 should be loc to idx=2(which is idx of old id=2), so change id=5 to 2. 
        # node_mapping = 1:1, 5:2, 7:5, ...

        parted_ids = [(p,node_ids[old_node_idx]) for old_node_idx, p in enumerate(parts)]
        parted_ids = [(p,old_node_id) for p, old_node_id in zip(parts, node_ids)]
        sorted_node_ids = [node_id for p, node_id in sorted(parted_ids)]

        id_mapping = dict((old_id, new_id) for old_id, new_id in zip(sorted_node_ids, node_ids))
        # new_id_dict = dict((old_node_id, i) for i,(p,old_node_id) in enumerate(parted_index)) squeeze ids
        new_edge_index = torch.clone(pyg_graph.edge_index)

        # to be tensorified
        pyg_graph.edge_index[0]= torch.tensor([id_mapping[old_id.item()] for old_id in new_edge_index[0]])
        pyg_graph.edge_index[1]= torch.tensor([id_mapping[old_id.item()] for old_id in new_edge_index[1]])

        # TODO features and masks
        pass

    def process_with_partition(self):
        print('reprocess with partition begin')
        limit_size = 50
        data = np.load(osp.join(self.raw_dir, 'reddit_data.npz'))
        x = torch.from_numpy(data['feature']).to(torch.float)[:limit_size]
        y = torch.from_numpy(data['label']).to(torch.long)[:limit_size]
        split = torch.from_numpy(data['node_types'])[:limit_size]

        print('features loaded')

        adj = sp.load_npz(osp.join(self.raw_dir, 'reddit_graph.npz'))
        row = torch.from_numpy(adj.row[adj.row<limit_size]).to(torch.long)
        col = torch.from_numpy(adj.col[adj.col<limit_size]).to(torch.long)

        edge_index = torch.stack([row, col], dim=0)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        print('edges loaded')

        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = split == 1
        data.val_mask = split == 2
        data.test_mask = split == 3

        edge_index = self.partition(data)

        data = data if self.pre_transform is None else self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def check_partition_connection(data, GPU=8):
    part_size = data.num_nodes//GPU
    all_nodes = torch.tensor(range(data.num_nodes))
    parts_node = list()
    for i in range(GPU):
        parts_node.append(all_nodes[i*part_size:(i+1)*part_size])

    self_edges = dict()
    for i in range(GPU):
        part_node = parts_node[i]
        part_edge_index, part_feat = pyg.utils.subgraph(part_node, data.edge_index)
        self_edges[i] = part_edge_index.size()[1]
        print('part  self', part_edge_index.size())

    for i in range(GPU):
        print('from',i)
        for j in range(GPU):
            if j<=i:
                continue
            two_node = torch.cat( (parts_node[i],  parts_node[j]) )
            two_edge_index, part_feat = pyg.utils.subgraph(two_node, data.edge_index)
            print('with', j, 'edges', two_edge_index.size()[1]-self_edges[i]-self_edges[j])


def main():
    device = torch.device('cuda', 5)

    begin = dt.datetime.now()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'SmallerReddit')
    dataset = SmallerReddit(path)
    data = dataset[0]

    load_t = dt.datetime.now()
    print('data loading time:', load_t-begin)

    print('train nodes amount:', sum(data.train_mask))
    print('test nodes amount:', sum(data.test_mask))
    print('val nodes amount:', sum(data.val_mask))
    print('features:', data.x.size())
    print('y:', data.y.size())
    print('edges :', data.edge_index.size())

    eval_t = dt.datetime.now()
    print('data eval time:', eval_t-load_t)

    torch.cuda.synchronize()
    x = data.x.to(device)
    y = data.y.to(device)
    e = data.edge_index.to(device)
    torch.cuda.synchronize()

    to_t = dt.datetime.now()
    print('data to gpu time:', to_t-eval_t)

    end = dt.datetime.now()
    print('total time:', end-begin)


if __name__=='__main__':
    main()
