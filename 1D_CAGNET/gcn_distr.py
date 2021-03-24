import os
import math
import time
import torch
import socket
import argparse
import statistics
import numpy as np
import os.path as osp
import datetime as dt
from collections import defaultdict

import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Manager, Process
from torch.nn import Parameter

import dist_data_util
from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu


total_time = defaultdict(float)
comp_time = defaultdict(float)
comp_localA_time = defaultdict(float)
comp_sep_times = defaultdict(lambda: defaultdict(float))
comm_time = defaultdict(float)
scomp_time = defaultdict(float)
dcomp_time = defaultdict(float)
bcast_comm_time = defaultdict(float)
barrier_time = defaultdict(float)
op1_comm_time = defaultdict(float)
op2_comm_time = defaultdict(float)

def dist_log(*args, group=None, rank=0):
    assert(g_rank>=0)
    if group is None:
        print(dt.datetime.now(), '[Rank %2d] '%g_rank, end='')
        print(*args, flush=True)
        with open('all_log_%d.txt'%g_rank, 'a+') as f:
            print(dt.datetime.now(), '[Rank %2d] '%g_rank, end='', file=f)
            print(*args, file=f, flush=True)
    else:
        assert(group is not None)
        dist.barrier(group)
        if g_rank==rank:
            print(dt.datetime.now(), '[Rank all] ', end='')
            print(*args)

def barrier_all():
    global barrier_time
    barrier_tstart = time.time()
    dist.barrier(g_group)
    barrier_tstop = time.time()
    barrier_time[g_rank] += barrier_tstop - barrier_tstart


def start_time(src=None):
    if not args.timing:
        return 0.0
    tstart = time.time()
    return tstart

def stop_time(tstart):
    if not args.timing:
        return 0.0
    tstop = time.time()
    return tstop - tstart


def block_row(adj_matrix, am_partitions, inputs, weight):
    n_per_proc = math.ceil(float(adj_matrix.size(1)) / g_world_size)

    z_loc = torch.cuda.FloatTensor(n_per_proc, inputs.size(1)).fill_(0)
    # z_loc = torch.zeros(adj_matrix.size(0), inputs.size(1))
    
    inputs_recv = torch.zeros(inputs.size())

    part_id = g_rank % g_world_size

    z_loc += torch.mm(am_partitions[part_id].t(), inputs) 

    for i in range(1, g_world_size):
        part_id = (g_rank + i) % g_world_size

        inputs_recv = torch.zeros(am_partitions[part_id].size(0), inputs.size(1))

        src = (g_rank + 1) % g_world_size
        dst = g_rank - 1
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

def outer_product2(inputs, ag):
    tstart_comp = start_time()
    # (H^(l-1))^T * (A * G^l)
    grad_weight = torch.mm(inputs, ag)
    dur = stop_time(tstart_comp)
    comp_time[g_rank] += dur
    dcomp_time[g_rank] += dur
    
    tstart_comm = start_time()
    # reduction on grad_weight low-rank matrices
    dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=g_group)
    barrier_all()
    dur = stop_time(tstart_comm)
    comm_time[g_rank] += dur
    op2_comm_time[g_rank] += dur

    return grad_weight



def broad_func(node_count, am_partitions, inputs):
    dist_log('broadcast begin')
    n_per_proc = math.ceil(float(node_count) / g_world_size)
    z_loc = torch.cuda.FloatTensor(am_partitions[0].size(0), inputs.size(1), device=device).fill_(0)
    inputs_recv = torch.cuda.FloatTensor(n_per_proc, inputs.size(1), device=device).fill_(0)
    for i in range(g_world_size):
        dist_log('loop', i, 'begin')
        if i == g_rank:
            dist_log('to clone sender data', inputs.size())

            ii = torch.ones(10, device=device)
            dist_log('torch OK', ii)
            time.sleep(1)
            i2 = inputs.clone()
            dist_log('clone OK')
            time.sleep(1)

            inputs_recv = None
            dist_log('recv clear OK')
            inputs_recv = inputs.clone()
            dist_log('sender data ready')
        elif i == g_world_size - 1:
            inputs_recv = torch.cuda.FloatTensor(am_partitions[i].size(1), inputs.size(1), device=device).fill_(0)
            dist_log('sender data ready')
            # inputs_recv = torch.zeros(list(am_partitions[i].t().size())[1], inputs.size(1))

        dist_log('before barrier', i)
        # barrier_all()
        dist_log(i,'bcast with sizes:','recv',inputs_recv.size(), 'inputs', inputs.size())

        tstart_comm = start_time()
        dist.broadcast(inputs_recv, src=i, group=g_group)
        barrier_all()
        dist_log('bcast done', i)
        dur = stop_time(tstart_comm)
        comm_time[g_rank] += dur
        bcast_comm_time[g_rank] += dur

        tstart_comp = start_time()
        tstart = time.time()
        dist_log(i, 'spmm with size',i, 'adj', am_partitions[i].size(), 
                                'zloc',z_loc.size(), 'recv',inputs_recv.size()) 
        spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                        am_partitions[i].values(), am_partitions[i].size(0), 
                        am_partitions[i].size(1), inputs_recv, z_loc)
        dist_log('spmm done',i) 
        tstop = time.time() 
        dur = stop_time( tstart_comp) 
        if i== g_rank: 
            comp_localA_time[g_rank] += dur 
        comp_sep_times[g_rank][i] += (tstop-tstart) 
        comp_time[g_rank] += dur 
        scomp_time[g_rank] += dur 
        dist_log('loop', i, 'end')
    dist_log('broadcast end')
    return z_loc 


class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, am_partitions, func):
        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.am_partitions = am_partitions
        ctx.func = func
        # z = block_row(adj_matrix.t(), am_partitions, inputs, weight, rank, size)
        z = broad_func(adj_matrix.size(0), am_partitions, inputs)
        dist_log('first bcast', group=g_group)

        tstart_comp = start_time()
        mid_feature = torch.mm(z, weight)

        dur = stop_time(tstart_comp)
        comp_time[g_rank] += dur
        dcomp_time[g_rank] += dur

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
        ag = broad_func(adj_matrix.size(0), am_partitions, grad_output)
        dist_log('second bcast', group=g_group)

        tstart_comp = start_time()
        grad_input = torch.mm(ag, weight.t())
        dur = stop_time(tstart_comp)
        comp_time[g_rank] += dur
        dcomp_time[g_rank] += dur
        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = outer_product2(inputs.t(), ag)
        return grad_input, grad_weight, None, None, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, am_partitions, optimizer):
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, am_partitions,  F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, am_partitions,  F.log_softmax)
    dist_log('layer forward done')

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
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    if len(accs) == 1:
        accs.append(0)
        accs.append(0)
    return accs


#from dist_start import g_rank, g_world_size
# def dist_main(tg_rank, tg_world_size, A_blocks, A_block_seps, H_blocks, labels, num_classes, args):
g_rank = -1
g_group = None
g_world_size = -1
device = None
args = None

def dist_main(ddevice, rank, world_size, world_group, p2p_group_dict, A_index, A_values, H, labels, num_classes, argss):
    global g_rank, g_world_size, g_group, device, args
    g_rank = rank
    g_world_size = world_size
    device = ddevice
    g_group = world_group
    args = argss

    # dist_log('in distmain', group=g_group)

    Ai, Ai_blocks, Hi = dist_data_util.partition_1D(A_index, A_values, H, args.nprocs, rank=g_rank)
    dist_log('data parted', Ai.size(), Hi.size())
    # return
    best_val_acc = test_acc = 0
    outputs = None

    inputs_loc, adj_matrix_loc, am_pbyp = Hi, Ai, Ai_blocks

    inputs_loc = inputs_loc.to(device)
    adj_matrix_loc = adj_matrix_loc.to(device)
    for i in range(len(am_pbyp)):
        am_pbyp[i] = am_pbyp[i].t().coalesce().to(device)

    torch.manual_seed(0)
    weight1_nonleaf = torch.rand(H.size(1), args.mid_layer, requires_grad=True, device=device)
    # weight1_nonleaf = weight1_nonleaf.to(device)
    weight1_nonleaf.retain_grad()

    weight2_nonleaf = torch.rand(args.mid_layer, num_classes, requires_grad=True)
    weight2_nonleaf = weight2_nonleaf.to(device)
    weight2_nonleaf.retain_grad()

    weight1 = Parameter(weight1_nonleaf)
    weight2 = Parameter(weight2_nonleaf)

    optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)
    dist.barrier(g_group)

    tstart = 0.0
    tstop = 0.0

    dist_log('ready to train', group=g_group)

    outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer)

    dist.barrier(g_group)
    tstart = time.time()

    # for epoch in range(1, 201):
    print(f"Starting training... rank {rank}", flush=True)
    for epoch in range(1, args.epochs):
        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer)
        print("Epoch: {:03d}".format(epoch), flush=True)

    tstop = time.time()
    total_time[g_rank] = tstop - tstart

    # Get median runtime according to rank0 and print that run's breakdown
    dist.barrier(g_group)
        
    # with open('m1d_r%d_%d_epochs.txt'%(g_rank, args.epochs), 'wt') as f:
    dist_log(f"rank: {rank} total_time: {total_time[rank]}")
    dist_log(f"rank: {rank} comm_time: {comm_time[rank]}")
    dist_log(f"rank: {rank} comp_time: {comp_time[rank]}")
    dist_log(f"rank: {rank} comp_localA_time: {comp_localA_time[rank]}")
    dist_log(f"rank: {rank} comp_sep_times: {comp_sep_times[rank]}")
    dist_log(f"rank: {rank} scomp_time: {scomp_time[rank]}")
    dist_log(f"rank: {rank} dcomp_time: {dcomp_time[rank]}")
    dist_log(f"rank: {rank} bcast_comm_time: {bcast_comm_time[rank]}")
    dist_log(f"rank: {rank} barrier_time: {barrier_time[rank]}")
    dist_log(f"rank: {rank} op1_comm_time: {op1_comm_time[rank]}")
    dist_log(f"rank: {rank} op2_comm_time: {op2_comm_time[rank]}") 
    
    if args.accuracy:
        # All-gather outputs to test accuracy
        output_parts = []
        n_per_proc = math.ceil(float(inputs.size(0)) / g_world_size)
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



if __name__ == '__main__':
    pass
