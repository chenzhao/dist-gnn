import os
import math
import time
import torch
import socket
import argparse
import statistics
import numpy as np
import os.path as osp
from collections import defaultdict

import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Manager, Process
from torch.nn import Parameter

import torch_sparse
from torch_scatter import scatter_add

import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, dense_to_sparse, to_scipy_sparse_matrix

from reddit import Reddit
from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

def distlog(*l):
    print(l)


total_time = defaultdict(float)
comp_time = defaultdict(float)
comp_localA_time = defaultdict(float)
comp_sep_times = defaultdict(lambda: defaultdict(float))
comm_time = defaultdict(float)
scomp_time = defaultdict(float)
dcomp_time = defaultdict(float)
bcast_comm_time = defaultdict(float)
barrier_time = defaultdict(float)
barrier_subset_time = defaultdict(float)
op1_comm_time = defaultdict(float)
op2_comm_time = defaultdict(float)

args = None
device = None
rank = -1
world_size = -1 

def start_time(group, rank, subset=False, src=None):
    global barrier_time
    global barrier_subset_time

    if not args.timing:
        return 0.0
    
    barrier_tstart = time.time()

    dist.barrier(group)
    barrier_tstop = time.time()
    barrier_time[rank] += barrier_tstop - barrier_tstart
    if subset:
        barrier_subset_time[rank] += barrier_tstop - barrier_tstart

    tstart = 0.0
    tstart = time.time()
    return tstart

def stop_time(group, rank, tstart):
    global barrier_time
    if not args.timing:
        return 0.0
    barrier_tstart = time.time()
    dist.barrier(group)
    barrier_tstop = time.time()
    barrier_time[rank] += barrier_tstop - barrier_tstart

    devid = rank_to_devid(rank, args.acc_per_rank)
    device = torch.device('cuda:{}'.format(devid))
    # torch.cuda.synchronize(device=device)

    tstop = time.time()
    return tstop - tstart


def normalize(adj_matrix):
    adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0))
    d = torch.sum(adj_matrix, dim=1)
    d = torch.rsqrt(d)
    d = torch.diag(d)
    return torch.mm(d, torch.mm(adj_matrix, d))

def block_row(adj_matrix, am_partitions, inputs, weight, rank, size):
    n_per_proc = math.ceil(float(adj_matrix.size(1)) / size)
    # n_per_proc = int(adj_matrix.size(1) / size)
    # am_partitions = list(torch.split(adj_matrix, n_per_proc, dim=1))

    z_loc = torch.cuda.FloatTensor(n_per_proc, inputs.size(1)).fill_(0)
    # z_loc = torch.zeros(adj_matrix.size(0), inputs.size(1))
    
    inputs_recv = torch.zeros(inputs.size())

    part_id = rank % size

    z_loc += torch.mm(am_partitions[part_id].t(), inputs) 

    for i in range(1, size):
        part_id = (rank + i) % size

        inputs_recv = torch.zeros(am_partitions[part_id].size(0), inputs.size(1))

        src = (rank + 1) % size
        dst = rank - 1
        if dst < 0:
            dst = size - 1

        if rank == 0:
            dist.send(tensor=inputs, dst=dst)
            dist.recv(tensor=inputs_recv, src=src)
        else:
            dist.recv(tensor=inputs_recv, src=src)
            dist.send(tensor=inputs, dst=dst)
        
        inputs = inputs_recv.clone()

        # z_loc += torch.mm(am_partitions[part_id], inputs) 
        z_loc += torch.mm(am_partitions[part_id].t(), inputs) 


    # z_loc = torch.mm(z_loc, weight)
    return z_loc

def outer_product(adj_matrix, grad_output, rank, size, group):
    n_per_proc = math.ceil(float(adj_matrix.size(0)) / size)
    
    tstart_comp = start_time(group, rank)

    # A * G^l
    ag = torch.mm(adj_matrix, grad_output)

    dur = stop_time(group, rank, tstart_comp)
    comp_time[rank] += dur
    dcomp_time[rank] += dur

    tstart_comm = start_time(group, rank)

    # reduction on A * G^l low-rank matrices
    dist.all_reduce(ag, op=dist.ReduceOp.SUM, group=group)

    dur = stop_time(group, rank, tstart_comm)
    comm_time[rank] += dur
    op1_comm_time[rank] += dur

    # partition A * G^l by block rows and get block row for this process
    # TODO: this might not be space-efficient
    red_partitions = list(torch.split(ag, n_per_proc, dim=0))
    grad_input = red_partitions[rank]

    return grad_input

def outer_product2(inputs, ag, rank, size, group):

    tstart_comp = start_time(group, rank)
    # (H^(l-1))^T * (A * G^l)
    grad_weight = torch.mm(inputs, ag)

    dur = stop_time(group, rank, tstart_comp)
    comp_time[rank] += dur
    dcomp_time[rank] += dur
    
    tstart_comm = start_time(group, rank)
    # reduction on grad_weight low-rank matrices
    dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=group)

    dur = stop_time(group, rank, tstart_comm)
    comm_time[rank] += dur
    op2_comm_time[rank] += dur

    return grad_weight



def broad_func(node_count, am_partitions, inputs, rank, size, group):
    # n_per_proc = math.ceil(float(adj_matrix.size(1)) / size)
    n_per_proc = math.ceil(float(node_count) / size)

    # z_loc = torch.cuda.FloatTensor(adj_matrix.size(0), inputs.size(1), device=device).fill_(0)
    z_loc = torch.cuda.FloatTensor(am_partitions[0].size(0), inputs.size(1), device=device).fill_(0)
    # z_loc = torch.zeros(adj_matrix.size(0), inputs.size(1))
    
    inputs_recv = torch.cuda.FloatTensor(n_per_proc, inputs.size(1), device=device).fill_(0)
    # inputs_recv = torch.zeros(n_per_proc, inputs.size(1))

    for i in range(size):
        if i == rank:
            inputs_recv = inputs.clone()
        elif i == size - 1:
            inputs_recv = torch.cuda.FloatTensor(am_partitions[i].size(1), inputs.size(1), device=device).fill_(0)
            # inputs_recv = torch.zeros(list(am_partitions[i].t().size())[1], inputs.size(1))

        tstart_comm = start_time(group, rank)

        dist.broadcast(inputs_recv, src=i, group=group)

        dur = stop_time(group, rank, tstart_comm)
        comm_time[rank] += dur
        bcast_comm_time[rank] += dur

        tstart_comp = start_time(group, rank)

        # print('rank', rank,'+++am parititions',i, am_partitions[i].values().size(), flush=True)

        # print('---A:', am_partitions[i].size())
        # print('---B:', inputs_recv.size())
        # print('---z_loc:', z_loc.size())
        tstart = time.time()

        spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                        am_partitions[i].values(), am_partitions[i].size(0), 
                        am_partitions[i].size(1), inputs_recv, z_loc)
        # print('z_loc after spmm:', z_loc.requires_grad, z_loc.size())
        # torch_sparse.spmm(am_partitions[i].indices(), 
        # am_partitions[i].values(), am_partitions[i].size(0), 
        # am_partitions[i].size(1), inputs_recv, z_loc)

        tstop = time.time()
        dur = stop_time(group, rank, tstart_comp)

        if i== rank:
            comp_localA_time[rank] += dur

        comp_sep_times[rank][i] += (tstop-tstart)

        comp_time[rank] += dur
        scomp_time[rank] += dur

    return z_loc

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, am_partitions, rank, size, group, func):
        # print('~~~~before fwd, inputs:', inputs.is_leaf, inputs.requires_grad, inputs.grad_fn, func)

        # inputs: H
        # adj_matrix: A
        # weight: W
        # func: sigma

        # adj_matrix = adj_matrix.to_dense()
        ctx.save_for_backward(inputs, weight, adj_matrix)
        # ctx.save_for_backward(inputs, weight, adj_matrix)
        # ctx.weight = weight
        # ctx.adj_matrix = adj_matrix

        ctx.am_partitions = am_partitions
        ctx.rank = rank
        ctx.size = size
        ctx.group = group

        ctx.func = func

        # z = block_row(adj_matrix.t(), am_partitions, inputs, weight, rank, size)
        z = broad_func(adj_matrix.size(0), am_partitions, inputs, rank, size, group)

        tstart_comp = start_time(group, rank)
        mid_feature = torch.mm(z, weight)

        dur = stop_time(group, rank, tstart_comp)
        comp_time[rank] += dur
        dcomp_time[rank] += dur

        mid_feature.requires_grad = True
        ctx.mid_feature = mid_feature

        if func is F.log_softmax:
            h = func(mid_feature, dim=1)
        elif func is F.relu:
            h = func(mid_feature)
        else:
            h = mid_feature

        return h

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, adj_matrix = ctx.saved_tensors

        am_partitions = ctx.am_partitions
        rank = ctx.rank
        size = ctx.size
        group = ctx.group

        func = ctx.func
        mid_feature = ctx.mid_feature

        with torch.set_grad_enabled(True):
            if func is F.log_softmax:
                func_eval = func(mid_feature, dim=1)
            elif func is F.relu:
                func_eval = func(mid_feature)
            else:
                func_eval = mid_feature

            sigmap = torch.autograd.grad(outputs=func_eval, inputs=mid_feature, grad_outputs=grad_output)[0]
            grad_output = sigmap

        # First backprop equation
        ag = broad_func(adj_matrix.size(0), am_partitions, grad_output, rank, size, group)

        tstart_comp = start_time(group, rank)

        grad_input = torch.mm(ag, weight.t())

        dur = stop_time(group, rank, tstart_comp)
        comp_time[rank] += dur
        dcomp_time[rank] += dur

        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = outer_product2(inputs.t(), ag, rank, size, group)

        return grad_input, grad_weight, None, None, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, am_partitions, optimizer, data, rank, size, group):
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, am_partitions, rank, size, group, F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, am_partitions, rank, size, group, F.log_softmax)

    optimizer.zero_grad()
    rank_train_mask = torch.split(data.train_mask.bool(), outputs.size(0), dim=0)[rank]
    datay_rank = torch.split(data.y, outputs.size(0), dim=0)[rank]

    # Note: bool type removes warnings, unsure of perf penalty
    # loss = F.nll_loss(outputs[data.train_mask.bool()], data.y[data.train_mask.bool()])
    if list(datay_rank[rank_train_mask].size())[0] > 0:
    # if datay_rank.size(0) > 0:
        loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask])
        # loss = F.nll_loss(outputs, torch.max(datay_rank, 1)[1])
        loss.backward()
    else:
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size(), device=device).fill_(0)).sum()
        # fake_loss = (outputs * torch.zeros(outputs.size())).sum()
        fake_loss.backward()

    optimizer.step()

    return outputs

def test(outputs, data, vertex_count, rank):
    logits, accs = outputs, []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    if len(accs) == 1:
        accs.append(0)
        accs.append(0)

    return accs
    # logits, accs = outputs, []
    # datay_rank = torch.split(data.y, vertex_count)[rank]
    # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
    #     mask_rank = torch.split(mask, vertex_count)[rank]
    #     count = mask_rank.nonzero().size(0)
    #     if count > 0:
    #         pred = logits[mask_rank].max(1)[1]
    #         acc = pred.eq(datay_rank[mask_rank]).sum().item() / mask_rank.sum().item()
    #         # pred = logits[mask].max(1)[1]
    #         # acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    #     else:
    #         acc = -1
    #     accs.append(acc)
    # return accs




def dist_main(g_rank, g_world_size, coo_adj_matrix, features, labels, num_classes, args):
# def run(rank, size, inputs, adj_matrix, data, features, classes, device):
    best_val_acc = test_acc = 0
    outputs = None

    inputs_loc, adj_matrix_loc, am_pbyp = oned_partition(rank, size, inputs, adj_matrix, data, 
                                                                features, classes, device)

    inputs_loc = inputs_loc.to(device)
    adj_matrix_loc = adj_matrix_loc.to(device)
    for i in range(len(am_pbyp)):
        am_pbyp[i] = am_pbyp[i].t().coalesce().to(device)

    torch.manual_seed(0)
    weight1_nonleaf = torch.rand(features, args.mid_layer, requires_grad=True)
    weight1_nonleaf = weight1_nonleaf.to(device)
    weight1_nonleaf.retain_grad()

    weight2_nonleaf = torch.rand(args.mid_layer, classes, requires_grad=True)
    weight2_nonleaf = weight2_nonleaf.to(device)
    weight2_nonleaf.retain_grad()

    weight1 = Parameter(weight1_nonleaf)
    weight2 = Parameter(weight2_nonleaf)

    optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)
    dist.barrier(group)

    tstart = 0.0
    tstop = 0.0

    outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, data, 
                            rank, size, group)

    dist.barrier(group)
    tstart = time.time()

    # for epoch in range(1, 201):
    print(f"Starting training... rank {rank}", flush=True)
    for epoch in range(1, args.epochs):
        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, data, 
                                rank, size, group)
        print("Epoch: {:03d}".format(epoch), flush=True)

    # dist.barrier(group)
    tstop = time.time()
    total_time[rank] = tstop - tstart

    # Get median runtime according to rank0 and print that run's breakdown
    dist.barrier(group)

        
    with open('m1d_r%d_%d_epochs.txt'%(rank, args.epochs), 'wt') as f:
        print(f"rank: {rank} total_time: {total_time[rank]}", file=f)
        print(f"rank: {rank} comm_time: {comm_time[rank]}", file=f)
        print(f"rank: {rank} comp_time: {comp_time[rank]}", file=f)
        print(f"rank: {rank} comp_localA_time: {comp_localA_time[rank]}", file=f)
        print(f"rank: {rank} comp_sep_times: {comp_sep_times[rank]}", file=f)
        print(f"rank: {rank} scomp_time: {scomp_time[rank]}", file=f)
        print(f"rank: {rank} dcomp_time: {dcomp_time[rank]}", file=f)
        print(f"rank: {rank} bcast_comm_time: {bcast_comm_time[rank]}", file=f)
        print(f"rank: {rank} barrier_time: {barrier_time[rank]}", file=f)
        print(f"rank: {rank} barrier_subset_time: {barrier_subset_time[rank]}", file=f)
        print(f"rank: {rank} op1_comm_time: {op1_comm_time[rank]}", file=f)
        print(f"rank: {rank} op2_comm_time: {op2_comm_time[rank]}", file=f)
        # print(f"rank: {rank} {outputs}")
    if rank==1:
        print(f"rank: {rank} total_time: {total_time[rank]}")
        print(f"rank: {rank} comm_time: {comm_time[rank]}")
        print(f"rank: {rank} comp_time: {comp_time[rank]}")
        print(f"rank: {rank} comp_localA_time: {comp_localA_time[rank]}")
        print(f"rank: {rank} comp_sep_times: {comp_sep_times[rank]}")
        print(f"rank: {rank} scomp_time: {scomp_time[rank]}")
        print(f"rank: {rank} dcomp_time: {dcomp_time[rank]}")
        print(f"rank: {rank} bcast_comm_time: {bcast_comm_time[rank]}")
        print(f"rank: {rank} barrier_time: {barrier_time[rank]}")
        print(f"rank: {rank} barrier_subset_time: {barrier_subset_time[rank]}")
        print(f"rank: {rank} op1_comm_time: {op1_comm_time[rank]}")
        print(f"rank: {rank} op2_comm_time: {op2_comm_time[rank]}")
        # print(f"rank: {rank} {outputs}")
    
    
    if args.accuracy:
        # All-gather outputs to test accuracy
        output_parts = []
        n_per_proc = math.ceil(float(inputs.size(0)) / size)
        # print(f"rows: {am_pbyp[-1].size(0)} cols: {classes}", flush=True)
        for i in range(size):
            output_parts.append(torch.cuda.FloatTensor(n_per_proc, classes, device=device).fill_(0))

        if outputs.size(0) != n_per_proc:
            pad_row = n_per_proc - outputs.size(0) 
            outputs = torch.cat((outputs, torch.cuda.FloatTensor(pad_row, classes, device=device)), dim=0)

        dist.all_gather(output_parts, outputs)
        output_parts[rank] = outputs
        
        padding = inputs.size(0) - n_per_proc * (size - 1)
        output_parts[size - 1] = output_parts[size - 1][:padding,:]

        outputs = torch.cat(output_parts, dim=0)

        train_acc, val_acc, tmp_test_acc = test(outputs, data, am_pbyp[0].size(1), rank)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc

        print('Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(train_acc, best_val_acc, test_acc))

    return outputs



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
def split_coo(adj_matrix, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    for i in range(len(vtx_indices) - 1):
        am_part = adj_matrix[:,(adj_matrix[dim,:] >= vtx_indices[i]).nonzero(as_tuple=False).squeeze(1)]
        am_part = am_part[:,(am_part[dim,:] < vtx_indices[i + 1]).nonzero(as_tuple=False).squeeze(1)]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

    return am_partitions, vtx_indices

# Normalize all elements according to KW's normalization rule
def scale_elements(adj_matrix, adj_part, node_count, row_vtx, col_vtx):
    if not args.normalization:
        return adj_part

    # Scale each edge (u, v) by 1 / (sqrt(u) * sqrt(v))
    # indices = adj_part._indices()
    # values = adj_part._values()

    # deg_map = dict()
    # for i in range(adj_part._nnz()):
    #     u = indices[0][i] + row_vtx
    #     v = indices[1][i] + col_vtx

    #     if u.item() in deg_map:
    #         degu = deg_map[u.item()]
    #     else:
    #         degu = (adj_matrix[0] == u).sum().item()
    #         deg_map[u.item()] = degu

    #     if v.item() in deg_map:
    #         degv = deg_map[v.item()]
    #     else:
    #         degv = (adj_matrix[0] == v).sum().item()
    #         deg_map[v.item()] = degv

    #     values[i] = values[i] / (math.sqrt(degu) * math.sqrt(degv))
    
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

def partition_1D(rank, size, inputs, adj_matrix, data, features, classes, device):
    node_count = inputs.size(0)
    n_per_proc = math.ceil(float(node_count) / size)

    am_partitions = None
    am_pbyp = None

    with torch.no_grad():
        # Column partitions
        am_partitions, vtx_indices = split_coo(adj_matrix, node_count, n_per_proc, 1)

        proc_node_count = vtx_indices[rank + 1] - vtx_indices[rank]
        am_pbyp, _ = split_coo(am_partitions[rank], node_count, n_per_proc, 0)
        for i in range(len(am_pbyp)):
            if i == size - 1:
                last_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)).to(device), 
                                                        size=(last_node_count, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank])
            else:
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], torch.ones(am_pbyp[i].size(1)).to(device), 
                                                        size=(n_per_proc, proc_node_count),
                                                        requires_grad=False)

                am_pbyp[i] = scale_elements(adj_matrix, am_pbyp[i], node_count, vtx_indices[i], 
                                                vtx_indices[rank])

        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], 
                                                    torch.ones(am_partitions[i].size(1)).to(device), 
                                                    size=(node_count, proc_node_count), 
                                                    requires_grad=False)
            am_partitions[i] = scale_elements(adj_matrix, am_partitions[i], node_count,  0, vtx_indices[i])

        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / size), dim=0)

        adj_matrix_loc = am_partitions[rank]
        inputs_loc = input_partitions[rank]
    inputs_loc.requires_grad_(False) # does not work on 1.7.1 for unkown reason

    inputs_loc = torch.tensor(inputs_loc, requires_grad = False)
    print(f"rank: {rank} adj_matrix_loc.size: {adj_matrix_loc.size()}", flush=True)
    print(f"rank: {rank} inputs.size: {inputs.size()}", flush=True)
    return inputs_loc, adj_matrix_loc, am_pbyp



if __name__ == '__main__':
    pass
