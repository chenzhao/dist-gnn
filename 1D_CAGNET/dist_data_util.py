import os
import math
import time
import torch
import numpy as np
import os.path as osp
from collections import defaultdict

import torch_sparse
from torch_scatter import scatter_add

from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, dense_to_sparse, to_scipy_sparse_matrix

from reddit import Reddit


def load_data(args):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.graphname)
    if args.graphname == "Cora":
        pyg_dataset = Planetoid(path, args.graphname, transform=T.NormalizeFeatures())
    elif args.graphname == "Reddit":
        pyg_dataset = Reddit(path, transform=T.NormalizeFeatures())
    pyg_data = pyg_dataset[0]
    features = pyg_data.x 
    labels = pyg_data.y
    coo_edge_index = pyg_data.edge_index
    num_classes = pyg_data.num_classes

    if args.normalization:
        coo_adj_matrix, _ = add_remaining_self_loops(coo_edge_index, num_nodes=pyg_data.num_nodes)
    else:
        coo_adj_matrix = coo_edge_index
    return coo_adj_matrix, features, labels, num_classes


# Split a COO into partitions of size n_per_proc
# Basically torch.split but for Sparse Tensors since pytorch doesn't support that.
def split_coo(coo_mat, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc)) + [node_count]

    coo_mat_parts = []
    for i in range(len(vtx_indices) - 1):
        part_mask = (coo_mat[dim, :]>=vtx_indices[i]) & (coo_mat[dim, :]<vtx_indices[i+1])
        coo_part = coo_mat[:, part_mask.nonzero(as_tuple=False).squeeze(1)]
        coo_part[dim] -= vtx_indices[i] # TODO: ?
        coo_mat_parts.append(coo_part)

    return coo_mat_parts, vtx_indices


# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx):
    if not args.normalization:
        return adj_part

    adj_part = adj_part.coalesce()
    deg = torch.histc(adj_matrix[0].double(), bins=node_count)
    deg = deg.pow(-0.5)

    row_len = adj_part.size(0)
    col_len = adj_part.size(1)

    dleft = torch.sparse_coo_tensor([np.arange(0, row_len).tolist(),
                                     np.arange(0, row_len).tolist()],
                                     deg[row_vtx:(row_vtx + row_len)].float(),
                                     size=(row_len, row_len),
                                     requires_grad=False, device=torch.device("cpu"))

    dright = torch.sparse_coo_tensor([np.arange(0, col_len).tolist(),
                                     np.arange(0, col_len).tolist()],
                                     deg[col_vtx:(col_vtx + col_len)].float(),
                                     size=(col_len, col_len),
                                     requires_grad=False, device=torch.device("cpu"))
    # adj_part = torch.sparse.mm(torch.sparse.mm(dleft, adj_part), dright)
    ad_ind, ad_val = torch_sparse.spspmm(adj_part._indices(), adj_part._values(), 
                                            dright._indices(), dright._values(),
                                            adj_part.size(0), adj_part.size(1), dright.size(1))

    adj_part_ind, adj_part_val = torch_sparse.spspmm(dleft._indices(), dleft._values(), 
                                                        ad_ind, ad_val,
                                                        dleft.size(0), dleft.size(1), adj_part.size(1))

    adj_part = torch.sparse_coo_tensor(adj_part_ind, adj_part_val, 
                                                size=(adj_part.size(0), adj_part.size(1)),
                                                requires_grad=False, device=torch.device("cpu"))
    return adj_part


def partition_1D(P, A, H):
    n = H.size(0)
    n_per_proc = math.ceil(n/P)
    A_blocks, node_indices = split_coo(A, n, n_per_proc, dim=1) # Column partitions
    A_block_seps = []
    for i in range(P):
        Ai_blocks, _ = split_coo(A_blocks[i], node_count, n_per_proc, dim=0)
        row_size = node_indices[i+1] - node_indices[i]
        for j in range(P):
            col_size = node_indices[j+1] - node_indices[j]
            coo_values = torch.ones(Ai_blocks[j].size(1))
            Ai_blocks[j] = torch.sparse_coo_tensor(Ai_blocks[j], coo_values, size=(row_size, col_size))
            # Ai_blocks[j] = scale_elements(A, Ai_blocks[j], node_count, vtx_indices[i], vtx_indices[rank])
        A_block_seps.append(Ai_blocks)

        coo_values = torch.ones(A_blocks[i].size(1))
        A_blocks[i] = torch.sparse_coo_tensor(A_blocks[i], coo_values, size=(n, row_size))
        # am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i])

    H_blocks = torch.split(H, n_per_proc, dim=0)

    # print(f"rank: {rank} A.size: {A.size()}", flush=True)
    # print(f"rank: {rank} inputs.size: {inputs.size()}", flush=True)
    return A_blocks, A_block_seps, H_blocks


if __name__ == '__main__':
    pass
