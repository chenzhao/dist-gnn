import os
import math
import time
import socket
import argparse
import statistics
import numpy as np
import os.path as osp
import datetime as dt

import torch
import torch.sparse
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn import Parameter
import torch.nn.functional as F

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu
from fast_reddit import Reddit, SmallerReddit

from collections import defaultdict
from dist_timer import DistTimer
import utils 

run = 0

total_time = defaultdict(lambda :defaultdict(float))
comp_time =defaultdict(lambda :defaultdict(float)) 
comm_time =defaultdict(lambda :defaultdict(float))
scomp_time =defaultdict(lambda :defaultdict(float))
dcomp_time =defaultdict(lambda :defaultdict(float)) 
bcast_comm_time =defaultdict(lambda :defaultdict(float)) 
barrier_time =defaultdict(lambda :defaultdict(float)) 
barrier_subset_time =defaultdict(lambda :defaultdict(float)) 
op1_comm_time =defaultdict(lambda :defaultdict(float))
op2_comm_time =defaultdict(lambda :defaultdict(float))


def outer_product2(inputs, ag):
    tstart_comp = start_time()
    # (H^(l-1))^T * (A * G^l)
    grad_weight = torch.mm(inputs, ag)

    dur = stop_time( tstart_comp)
    comp_time[run][g_rank] += dur
    dcomp_time[run][g_rank] += dur
    
    tstart_comm = start_time()
    # reduction on grad_weight low-g_rank matrices
    dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=g_world_group)

    dur = stop_time(tstart_comm)
    comm_time[run][g_rank] += dur
    op2_comm_time[run][g_rank] += dur

    return grad_weight

p2p_group_dict = None
def p2p_broad_func(node_count, am_partitions, inputs):
    # n_per_proc = math.ceil(float(adj_matrix.size(1)) / size)
    n_per_proc = math.ceil(float(node_count) / g_world_size)

    # z_loc = torch.cuda.FloatTensor(adj_matrix.size(0), inputs.size(1), device=device).fill_(0)
    z_loc = torch.cuda.FloatTensor(am_partitions[0].size(0), inputs.size(1), device=device).fill_(0)
    # z_loc = torch.zeros(adj_matrix.size(0), inputs.size(1))
    
    inputs_recv = torch.cuda.FloatTensor(n_per_proc, inputs.size(1), device=device).fill_(0)
    # inputs_recv = torch.zeros(n_per_proc, inputs.size(1))

    for i in range(g_world_size):
        if i == g_rank:
            inputs_recv = inputs.clone()
        elif i == g_world_size - 1:
            inputs_recv = torch.cuda.FloatTensor(am_partitions[i].size(1), inputs.size(1), device=device).fill_(0)
            # inputs_recv = torch.zeros(list(am_partitions[i].t().size())[1], inputs.size(1))
        barrier_all()

        tstart_comm = start_time()

        dist.broadcast(inputs_recv, src=i, group=g_world_group)

        dur = stop_time( tstart_comm)
        comm_time[run][g_rank] += dur
        bcast_comm_time[run][g_rank] += dur

        barrier_all()

        tstart_comp = start_time()

        spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                        am_partitions[i].values(), am_partitions[i].size(0), 
                        am_partitions[i].size(1), inputs_recv, z_loc)

        dur = stop_time( tstart_comp)
        comp_time[run][g_rank] += dur
        scomp_time[run][g_rank] += dur

    return z_loc

def broad_func(node_count, am_partitions, inputs):
    n_per_proc = math.ceil(float(node_count) / g_world_size)
    z_loc = torch.zeros((am_partitions[0].size(0), inputs.size(1)), device=device)
    inputs_recv = torch.zeros((n_per_proc, inputs.size(1)), device=device)

    for i in range(g_world_size):
        if i == g_rank:
            inputs_recv = inputs.clone()
        elif i == g_world_size - 1:
            inputs_recv = torch.zeros((am_partitions[i].size(1), inputs.size(1)), device=device)
        barrier_all()
        tstart_comm = start_time()
        dist.broadcast(inputs_recv, src=i, group=g_world_group)
        barrier_all()
        dur = stop_time( tstart_comm)
        comm_time[run][g_rank] += dur
        bcast_comm_time[run][g_rank] += dur

        tstart_comp = start_time()

        spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                        am_partitions[i].values(), am_partitions[i].size(0), 
                        am_partitions[i].size(1), inputs_recv, z_loc)

        dur = stop_time( tstart_comp)
        comp_time[run][g_rank] += dur
        scomp_time[run][g_rank] += dur

    return z_loc

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, am_partitions,  func):
        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.am_partitions = am_partitions
        ctx.func = func

        z = broad_func(adj_matrix.size(0), am_partitions, inputs)
        tstart_comp = start_time()
        z = torch.mm(z, weight)

        dur = stop_time( tstart_comp)
        comp_time[run][g_rank] += dur
        dcomp_time[run][g_rank] += dur

        z.requires_grad = True
        ctx.z = z

        if func is F.log_softmax:
            h = func(z, dim=1)
        elif func is F.relu:
            h = func(z)
        else:
            h = z
        return h

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, adj_matrix = ctx.saved_tensors
        am_partitions = ctx.am_partitions

        func = ctx.func
        z = ctx.z

        with torch.set_grad_enabled(True):
            if func is F.log_softmax:
                func_eval = func(z, dim=1)
            elif func is F.relu:
                func_eval = func(z)
            else:
                func_eval = z

            sigmap = torch.autograd.grad(outputs=func_eval, inputs=z, grad_outputs=grad_output)[0]
            grad_output = sigmap

        # First backprop equation
        ag = broad_func(adj_matrix.size(0), am_partitions, grad_output)

        tstart_comp = start_time()
        grad_input = torch.mm(ag, weight.t())
        dur = stop_time( tstart_comp)
        comp_time[run][g_rank] += dur
        dcomp_time[run][g_rank] += dur
        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = outer_product2(inputs.t(), ag)

        return grad_input, grad_weight, None, None, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, am_partitions, optimizer, data):
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, am_partitions, F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, am_partitions, F.log_softmax)

    optimizer.zero_grad()
    rank_train_mask = torch.split(data.train_mask.bool(), outputs.size(0), dim=0)[g_rank]
    datay_rank = torch.split(data.y, outputs.size(0), dim=0)[g_rank]

    if list(datay_rank[rank_train_mask].size())[0] > 0:
        loss = F.nll_loss(outputs[rank_train_mask], datay_rank[rank_train_mask])
        loss.backward()
    else:
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size(), device=device).fill_(0)).sum()
        fake_loss.backward()

    optimizer.step()
    return outputs

def test(outputs, data, vertex_count, rank):
    logits, accs = outputs, []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def split_coo_with_values(adj_matrix, adj_values, node_count, n_per_proc, dim):
    vtx_indices = list(range(0, node_count, n_per_proc))
    vtx_indices.append(node_count)

    am_partitions = []
    av_partitions = []
    for i in range(len(vtx_indices) - 1):
        mask = ((adj_matrix[dim,:] >= vtx_indices[i]) & (adj_matrix[dim,:] < vtx_indices[i + 1])).nonzero().squeeze(1)
        am_part = adj_matrix[:,mask]
        am_part[dim] -= vtx_indices[i]
        am_partitions.append(am_part)

        av_part = adj_values[mask]
        av_partitions.append(av_part)

    return am_partitions, av_partitions, vtx_indices

def oned_partition(inputs, adj_matrix, adj_values, data, features, classes):
    node_count = inputs.size(0)
    n_per_proc = math.ceil(float(node_count) / g_world_size)
    am_partitions = None
    am_pbyp = None

    with torch.no_grad():
        am_partitions, av_partitions, vtx_indices = split_coo_with_values(adj_matrix, adj_values, node_count, n_per_proc, 1)

        proc_node_count = vtx_indices[g_rank + 1] - vtx_indices[g_rank]
        am_pbyp, av_pbyp, _ = split_coo_with_values(am_partitions[g_rank], av_partitions[g_rank], node_count, n_per_proc, 0)
        for i in range(len(am_pbyp)):
            uni_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], av_pbyp[i], size=(uni_node_count, proc_node_count), requires_grad=False).coalesce()
        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], av_partitions[i], size=(node_count, proc_node_count), requires_grad=False).coalesce()
        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / g_world_size), dim=0)
    print('Rank',g_rank,'parted')
    return  input_partitions[g_rank], am_partitions[g_rank], am_pbyp

def main(inputs, adj_matrix, adj_values, data, features, classes):
    global run

    inputs_loc, adj_matrix_loc, am_pbyp = oned_partition(inputs, adj_matrix, adj_values, data, features, classes)
    inputs_loc = inputs_loc.to(device)  # to device
    adj_matrix_loc = adj_matrix_loc.to(device)
    for i in range(len(am_pbyp)):
        am_pbyp[i] = am_pbyp[i].t().coalesce().to(device)

    for i in range(args.run_count):
        run = i
        torch.manual_seed(0)
        weight1_nonleaf = torch.rand(features, args.mid_layer, requires_grad=True, device=device)
        weight1_nonleaf.retain_grad()

        weight2_nonleaf = torch.rand(args.mid_layer, classes, requires_grad=True, device=device)
        weight2_nonleaf.retain_grad()

        weight1 = Parameter(weight1_nonleaf)
        weight2 = Parameter(weight2_nonleaf)

        optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)
        dist.barrier(g_world_group)

        tstart = 0.0
        tstop = 0.0

        print(f"Starting training... rank {g_rank} run {i}", flush=True)
        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, data)

        dist.barrier(g_world_group)
        tstart = time.time()

        for epoch in range(1, args.epochs):
            outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, data)
            if epoch%10==0:
                print("Epoch: {:03d}".format(epoch), flush=True)

        tstop = time.time()
        total_time[i][g_rank] = tstop - tstart

    dist.barrier(g_world_group)
    if g_rank == 0:
        total_times_r0 = [] 
        for i in range(args.run_count):
            total_times_r0.append(total_time[i][0])

        print(f"total_times_r0: {total_times_r0}")
        median_run_time = statistics.median(total_times_r0)
        median_idx = total_times_r0.index(median_run_time)
        median_idx = torch.cuda.LongTensor([median_idx])
    else:
        median_idx = torch.cuda.LongTensor([0])
        
    dist.broadcast(median_idx, src=0, group=g_world_group)        
    median_idx = median_idx.item()
    with open('o1d%d_100_epochs.txt'%(g_rank), 'wt') as f:
        print(f"rank: {g_rank} median_run: {median_idx}", file=f)
        print(f"rank: {g_rank} total_time: {total_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} comm_time: {comm_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} comp_time: {comp_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} scomp_time: {scomp_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} dcomp_time: {dcomp_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} bcast_comm_time: {bcast_comm_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} barrier_time: {barrier_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} op1_comm_time: {op1_comm_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} op2_comm_time: {op2_comm_time[median_idx][g_rank]}", file=f)
        print(f"rank: {g_rank} {outputs}", file=f)
    
    output_parts = []
    n_per_proc = math.ceil(float(inputs.size(0)) / g_world_size)
    # print(f"rows: {am_pbyp[-1].size(0)} cols: {classes}", flush=True)
    for i in range(g_world_size):
        output_parts.append(torch.cuda.FloatTensor(n_per_proc, classes, device=device).fill_(0))

    if outputs.size(0) != n_per_proc:
        pad_row = n_per_proc - outputs.size(0) 
        outputs = torch.cat((outputs, torch.cuda.FloatTensor(pad_row, classes, device=device)), dim=0)

    dist.all_gather(output_parts, outputs)
    output_parts[g_rank] = outputs
    
    padding = inputs.size(0) - n_per_proc * (g_world_size - 1)
    output_parts[g_world_size - 1] = output_parts[g_world_size - 1][:padding,:]
    outputs = torch.cat(output_parts, dim=0)

    train_acc, val_acc, test_acc = test(outputs, data, am_pbyp[0].size(1), g_rank)
    print( 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(args.epochs, train_acc, val_acc, test_acc))
    return outputs

def rank_to_devid(g_rank):
    return g_rank

def prepare_and_run():
    if args.graphname == "Reddit":
        data = Reddit()
    elif args.graphname == "SmallerReddit":
        data = SmallerReddit()
    print('loaded to host mem')
    #inputs = data.x.to(device)
    #data.y = data.y.to(device)
    inputs = data.x
    adj_matrix = data.edge_index
    adj_values = data.normalized_adj_values
    num_features = data.x.size(1)
    num_classes = torch.unique(data.y).size(0)
    print(args.graphname, 'data loaded')
    main(inputs, adj_matrix, adj_values, data, num_features, num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run_count", type=int, default=1)
    parser.add_argument("--graphname", type=str, default="SmallerReddit")
    parser.add_argument("--timing", type=bool, default=True)
    parser.add_argument("--mid_layer", type=int, default=16)
    args = parser.parse_args()
    print(args)
    g_rank, g_world_size = args.local_rank, args.world_size
    device = torch.device('cuda:{}'.format(rank_to_devid(g_rank)))
    torch.cuda.set_device(device)

    dist.init_process_group(backend='nccl')
    g_world_group = dist.new_group(list(range(g_world_size)))
    timer=DistTimer(g_rank,g_world_group)
    p2p_begin = dt.datetime.now()
    p2p_group_dict = {}
    for src in range(g_world_size):
        for dst in range(src+1, g_world_size):
            p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
            p2p_group_dict[(dst, src)] = p2p_group_dict[(src, dst)]
    print('P2P groups inited', dt.datetime.now()-p2p_begin)
    utils.dist_log('test', p2p_begin)
    prepare_and_run()

