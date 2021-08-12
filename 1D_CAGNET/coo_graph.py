import os.path
import torch
import torch.sparse


def sym_lap(edge_index):
    uniques, inverse, counts = torch.unique(edge_index[0], sorted=True, return_inverse=True, return_counts=True)
    d_rsqrt = torch.rsqrt(counts.to(torch.double))
    edge_count = edge_index.size(1)
    sym_lap_values = torch.zeros(edge_count)
    prog_size = edge_count//100 or 1
    for i in range(edge_count):
        # d_u = d_rsqrt[inverse[i]]
        u = edge_index[0][i]
        v = edge_index[1][i]
        d_u = d_rsqrt[u]
        d_v = d_rsqrt[v]
        sym_lap_values[i] = d_u*d_v
        if i%prog_size==0:
            print('sym lap to 100:', i//prog_size, '\r', end='')
    print('\nsym lap done')
    return sym_lap_values


def add_self_loops(edge_index, num_nodes):
    r"""from pyg"""
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
    return edge_index


class COO_Graph:
    def torch_ready_data_file(self):
        return os.path.join('..', 'torch_ready_data', self.graph_name, 'processed', 'data.pt')
    def coo_graph_file(self):
        return os.path.join('..', 'coo_graph_data', self.graph_name+'.pt')
    attrs = ['DAD_coo', 'features', 'labels', 'adj_idx',
              'num_nodes', 'num_edges', 'num_features', 'num_classes',
              'train_mask', 'val_mask', 'test_mask']
    @property
    def DAD_idx(self): return self.DAD_coo._indices()
    @property
    def DAD_val(self): return self.DAD_coo._values()

    def __init__(self, graph_name):
        self.graph_name = graph_name
        if os.path.exists(self.coo_graph_file()):# and False:
            self.load_coo_file()
        else:
            self.load_torch_data_file()
            self.process_for_gcn()
            self.save_coo_file()

    def __repr__(self):
        train = torch.sum(self.train_mask)
        val = torch.sum(self.val_mask)
        test = torch.sum(self.test_mask)
        return f'{self.__class__.__name__} |V|: {self.num_nodes}, |E|: {self.num_edges}, |mask|: {train}, {val}, {test}, name: {self.graph_name}'

    def load_coo_file(self):
        d = torch.load(self.coo_graph_file())
        for attr in self.attrs:
            setattr(self, attr, d[attr])
        # fix wrong data
        if self.graph_name=='Yelp':
            print('Yelp detected')
            self.num_classes = 100
        elif self.graph_name=='AmazonProducts':
            self.num_classes = 107

    def save_coo_file(self):
        os.makedirs(os.path.dirname(self.coo_graph_file()), exist_ok=True)
        torch.save(dict((attr, getattr(self, attr)) for attr in self.attrs), self.coo_graph_file())

    def load_torch_data_file(self):
        d = torch.load(self.torch_ready_data_file())
        self.features = d['x']
        self.labels = d['y']
        self.adj_idx = d['edge_index']  # size=(2, |E|)

        self.num_nodes = d['x'].size(0)
        self.num_edges = d['edge_index'].size(1)
        self.num_features = d['x'].size(1)
        if self.graph_name=='Yelp':
            print('Yelp detected, assign num classes 100')
            self.num_classes = 100
        elif self.graph_name=='AmazonProducts':
            print('AmazonProducts detected, assign num classes 100')
            self.num_classes = 107
        else:
            print('unknown dataset detected, assign num classes by counting')
            self.num_classes = torch.unique(d['y']).size(0)  # may be predefined

        if d['split']:
            self.train_mask = d['split'] == 1
            self.val_mask   = d['split'] == 2
            self.test_mask  = d['split'] == 3
        elif d['mask']:
            self.train_mask, self.val_mask, self.test_mask  = d['mask']
        else:
            print('invalid torch data file')
            raise Exception()

    def process_for_gcn(self):  # make the coo format sym lap matrix
        DAD_idx = add_self_loops(self.adj_idx, self.num_nodes)
        DAD_val = sym_lap(DAD_idx)
        self.DAD_coo = torch.sparse_coo_tensor(DAD_idx, DAD_val, (self.num_nodes, self.num_nodes)).coalesce()
        self.features =  self.features / self.features.sum(1, keepdim=True).clamp(min=1)


import random
import numpy as np
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    torch.set_printoptions(precision=10)

    r = COO_Graph('Flickr')
    # r = COO_Graph('AmazonProducts')
    # r = COO_Graph('Yelp')
    print(r)
    return 

    # r = COO_Graph('Reddit')
    # pr = COO_Graph('PartedReddit')
    # r = COO_Graph('SmallerReddit')
    # pr = COO_Graph('PartedSmallerReddit')
    # r = COO_Graph('OneQuarterReddit')
    # pr = COO_Graph('PartedOneQuarterReddit')
    pass


if __name__ == '__main__':
    main()
