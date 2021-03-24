import os
import math
import time
import torch
import numpy as np
import os.path as osp
from collections import defaultdict

# from torch_geometric.datasets import Planetoid, PPI

from fast_reddit import Reddit, SmallerReddit


def load_data(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.graphname)
    if args.graphname == "Cora":
        pyg_dataset = Planetoid(path, args.graphname, transform=T.NormalizeFeatures())
    elif args.graphname == "Reddit":
        data = Reddit()
    elif args.graphname == "SmallerReddit":
        data = SmallerReddit()
        # pyg_dataset = Reddit(path, transform=T.NormalizeFeatures())
    features = data.x
    labels = data.y 
    coo_adj_mat = data.edge_index
    adj_values = data.normalized_adj_values
    num_features = data.x.size(1)
    num_classes = torch.unique(data.y).size(0)
    return coo_adj_mat, adj_values, features, labels, num_classes


def split_coo(coo_indices, coo_values, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc)) + [node_count]

    ind_parts, val_parts = [],[]
    for i in range(len(vtx_indices) - 1):
        part_mask = (coo_indices[dim, :]>=vtx_indices[i]) & (coo_indices[dim, :]<vtx_indices[i+1])
        ind_part = coo_indices[:, part_mask.nonzero(as_tuple=False).squeeze(1)]  # TODO: remove
        ind_part[dim] -= vtx_indices[i] 
        ind_parts.append(ind_part)
        val_parts.append(coo_values[part_mask.nonzero(as_tuple=False).squeeze(1)])  # TODO: remove

    return ind_parts, val_parts, vtx_indices


def partition_1D(A_indices, A_values, H, P, rank):
    i = rank
    n = H.size(0)
    n_per_proc = math.ceil(n/P)
    A_idx_blocks, A_val_blocks, node_indices = split_coo(A_indices, A_values, n, n_per_proc, dim=1) # Column partitions
    H_blocks = torch.split(H, n_per_proc, dim=0)
    Ai_blocks = []

    row_size = node_indices[i+1] - node_indices[i]
    Ai_idx_blocks, Ai_val_blocks, _ = split_coo(A_idx_blocks[i], A_val_blocks[i], n, n_per_proc, dim=0)
    for j in range(P):
        col_size = node_indices[j+1] - node_indices[j]
        print('rank', rank, 'j', j ,'size', col_size, row_size)
        Ai_blocks.append( torch.sparse_coo_tensor(Ai_idx_blocks[j], Ai_val_blocks[j], size=(col_size, row_size)).coalesce())

    A_block_i = torch.sparse_coo_tensor(A_idx_blocks[i], A_val_blocks[i], size=(n, row_size)).coalesce()
    print(rank, 'final size', A_block_i.size(),  H_blocks[i].size())
    return A_block_i, Ai_blocks, H_blocks[i]


if __name__ == '__main__':
    pass
