import math
import argparse
import datetime as dt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn import Parameter
import torch.nn.functional as F

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

from dist_timer import DistTimer
import utils 

run = 0

def outer_product2(inputs, ag):
    g_timer.start_time('mm')
    grad_weight = torch.mm(inputs, ag) # (H^(l-1))^T * (A * G^l)
    g_timer.stop_time('mm', 'comp')
    
    g_timer.start_time('all reduce')
    dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=g_env.world_group)
    g_timer.stop_time('all reduce', 'comm')
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
    device = g_env.device
    n_per_proc = math.ceil(float(node_count) / g_world_size)
    z_loc = torch.zeros((am_partitions[0].size(0), inputs.size(1)), device=device)
    inputs_recv = torch.zeros((n_per_proc, inputs.size(1)), device=device)

    for i in range(g_world_size):
        if i == g_rank:
            inputs_recv = inputs.clone()
        elif i == g_world_size - 1:
            inputs_recv = torch.zeros((am_partitions[i].size(1), inputs.size(1)), device=device)
        barrier_all()
        g_timer.start_time('broadcast')
        dist.broadcast(inputs_recv, src=i, group=g_env.world_group)
        g_timer.stop_time('comm')

        g_timer.start_time('spmm')
        spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                        am_partitions[i].values(), am_partitions[i].size(0), 
                        am_partitions[i].size(1), inputs_recv, z_loc)
        g_timer.start_time('spmm', 'comp')
    return z_loc

class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, am_partitions,  activation_func):
        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.am_partitions = am_partitions
        ctx.activation_func = activation_func

        z = broad_func(adj_matrix.size(0), am_partitions, inputs)

        g_timer.start_time('mm')
        z = torch.mm(z, weight)
        g_timer.start_time('mm', 'comp')

        z.requires_grad = True
        ctx.z = z
        return activation_func(z)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, adj_matrix = ctx.saved_tensors
        am_partitions = ctx.am_partitions

        with torch.set_grad_enabled(True):
            func_eval = ctx.activation_func(ctx.z)
            sigmap = torch.autograd.grad(outputs=func_eval, inputs=ctx.z, grad_outputs=grad_output)[0]
            grad_output = sigmap

        # First backprop equation
        ag = broad_func(adj_matrix.size(0), am_partitions, grad_output)

        g_timer.start_time('mm')
        grad_input = torch.mm(ag, weight.t())
        g_timer.stop('mm', 'comp')
        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = outer_product2(inputs.t(), ag)

        return grad_input, grad_weight, None, None, None, None, None, None

def train(inputs, weight1, weight2, adj_matrix, am_partitions, optimizer, data):
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, am_partitions, F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, am_partitions, lambda x:F.log_softmax(x, dim=1))

    optimizer.zero_grad()
    rank_train_mask = torch.split(data.train_mask.bool(), outputs.size(0), dim=0)[g_env.rank]
    datay_rank = torch.split(data.y, outputs.size(0), dim=0)[g_env.rank]

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

def main():
    global run
    inputs_loc, adj_matrix_loc, am_pbyp = g_data.local_features, g_data.local_adj, g_data.local_adj_parts 

    for i in range(args.run_count):
        run = i
        torch.manual_seed(0)
        weight1_nonleaf = torch.rand(features, args.mid_layer, requires_grad=True, device=g_env.device)
        weight1_nonleaf.retain_grad()

        weight2_nonleaf = torch.rand(args.mid_layer, classes, requires_grad=True, device=g_env.device)
        weight2_nonleaf.retain_grad()

        weight1 = Parameter(weight1_nonleaf)
        weight2 = Parameter(weight2_nonleaf)

        optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)
        # dist.barrier(g_world_group)

        print(f"Starting training... rank {g_rank} run {i}", flush=True)
        outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer)
        # dist.barrier(g_world_group)
        g_timer.start_time('train')
        for epoch in range(1, args.epochs):
            outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer)
            if epoch%10==0:
                print("Epoch: {:03d}".format(epoch), flush=True)
        g_timer.stop_time('train')

    n_per_proc = math.ceil(float(inputs.size(0)) / g_world_size)
    output_parts = [torch.zeros(n_per_proc, classes, device=g_env.device) for i in range(g_env.world_size)]

    if outputs.size(0) != n_per_proc:
        pad_row = n_per_proc - outputs.size(0) 
        outputs = torch.cat((outputs, torch.cuda.FloatTensor(pad_row, classes, device=g_env.device)), dim=0)

    dist.all_gather(output_parts, outputs)
    output_parts[g_rank] = outputs
    
    padding = inputs.size(0) - n_per_proc * (g_world_size - 1)
    output_parts[g_world_size - 1] = output_parts[g_world_size - 1][:padding,:]
    outputs = torch.cat(output_parts, dim=0)

    train_acc, val_acc, test_acc = test(outputs, data, am_pbyp[0].size(1), g_rank)
    print( 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(args.epochs, train_acc, val_acc, test_acc))
    return outputs


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
    g_env = utils.DistEnv(args.local_rank, args.world_size)

    g_timer = DistTimer(g_env)
    g_logger = utils.DistLogger(g_env)
    g_logger.log( 'dist env inited')

    g_data = utils.DistData(g_env, args.graphname)
    g_logger.log( 'dist data inited')

    main()

