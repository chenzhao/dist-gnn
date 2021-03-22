import os
import math
import datetime as dt
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch



def sym_lap(self):
# def slow_slow_scale(adj_matrix, adj_part, node_count, row_vtx, col_vtx):
    print('slow normalization begin')
    indices = self.edge_index
    values = torch.zeros(indices.size(1))

    deg_map = dict()
    for i in range(indices.size(1)):
        u = indices[0][i] 
        v = indices[1][i]
        if u.item() in deg_map:
            degu = deg_map[u.item()]
        else:
            degu = (indices[0] == u).sum().item()
            deg_map[u.item()] = degu
            if v.item() in deg_map:
                degv = deg_map[v.item()]
            else:
                degv = (indices[0] == v).sum().item()
                deg_map[v.item()] = degv
        values[i] = 1 / (math.sqrt(degu) * math.sqrt(degv))
    print('slow normalization end')
    return values


class SmallerReddit():
    def __init__(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'SmallerReddit','processed','data.pt.fast')
        d = torch.load(path)
        self.x = d['x']
        self.y = d['y']
        self.edge_index = d['edge_index']
        self.train_mask, self.val_mask, self.test_mask = d['masks']
        self.normalized_adj_values = d["sym_lap"]

        # {"x":d.x, "y":d.y, "edge_index":d.edge_index},


class Reddit():
    def __init__(self):
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit','processed','data.pt.fast')
        # self.data, self.slices = torch.load(path)
        d = torch.load(path)
        self.x = d['x']
        self.y = d['y']
        self.edge_index = d['edge_index']
        self.train_mask, self.val_mask, self.test_mask = d['masks']
        self.normalized_adj_values = d["sym_lap"]


def main():
    begin = dt.datetime.now()

    #path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'SmallerReddit')
    #dataset = SmallerReddit(path)
    data = SmallerReddit()

    print(data.x.size())
    print(torch.sum(data.x))
    return 

    # data = Reddit()

    load_t = dt.datetime.now()
    print('data loading time:', load_t-begin)

    device = torch.device('cuda', 7)
    torch.cuda.synchronize()

    # print('train nodes amount:', sum(data.train_mask))
    # print('test nodes amount:', sum(data.test_mask))
    # print('val nodes amount:', sum(data.val_mask))
    # print('features:', data.x.size())
    # print('y:', data.y.size())
    # print('edges :', data.edge_index.size())

    eval_t = dt.datetime.now()
    print('data eval time:', eval_t-load_t)

    torch.cuda.synchronize()
    # print('sizes:', data.x.size(), data.y.size(), data.edge_index.size())

    cuda_t = dt.datetime.now()
    print('cuda sync time:', cuda_t-eval_t)

    x = data.x.to(device)
    torch.cuda.synchronize()
    x_t = dt.datetime.now()
    print('x to gpu time:', x_t-cuda_t)

    y = data.y.to(device)
    torch.cuda.synchronize()
    y_t = dt.datetime.now()
    print('y to gpu time:', y_t-x_t)
    # print('y to gpu time:', y_t-x_t)

    x = data.x.to(device)
    torch.cuda.synchronize()
    x_t = dt.datetime.now()
    print('x to gpu time:', x_t-y_t)

    e = data.edge_index.to(device)
    torch.cuda.synchronize()
    e_t = dt.datetime.now()
    print('edge to gpu time:', e_t-y_t)

    torch.cuda.synchronize()
    end = dt.datetime.now()
    print('all data to gpu time:', end-cuda_t)

    print('total time:', end-begin)


if __name__=='__main__':
    main()
