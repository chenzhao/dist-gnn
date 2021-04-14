import os
import math
import argparse

import torch
import torch.distributed as dist

from torch.nn import Parameter
import torch.nn.functional as F

from sparse_coo_tensor_cpp import sparse_coo_tensor_gpu, spmm_gpu

import utils
from dist_data import DistData

run = 0

def outer_product2(inputs, ag):
    torch.cuda.synchronize()

    g_timer.start('mm')
    grad_weight = torch.mm(inputs, ag) # (H^(l-1))^T * (A * G^l)
    torch.cuda.synchronize()
    g_timer.stop('mm')#, 'comp')
    
    g_timer.start('all reduce')
    dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM, group=g_env.world_group)
    torch.cuda.synchronize()
    g_timer.stop('all reduce')#, 'comm')
    return grad_weight


def p2p_broadcast(t, src):
    for dst in range(g_env.world_size):
        if src==dst or g_env.rank not in (src, dst):
            # g_logger.log('p2p bcast skip', src, dst)
            continue
        dst_adj_nz_col = g_data.nz_col_dict[(dst, src)]  #  non zero
        needed_rows_idx = dst_adj_nz_col
        if g_env.rank==src:
            p2p_buf = t[needed_rows_idx]
        elif g_env.rank == dst:
            p2p_buf = torch.zeros((needed_rows_idx.size(0), t.size(1)), device=g_env.device)
        g_logger.log('p2p data ready', src, dst, 'needed size',p2p_buf.size(0), 'full size', t.size(0))
        dist.broadcast(p2p_buf, src, group=g_env.p2p_group_dict[(src, dst)])
        # g_logger.log('p2p bcast done', src, dst)
        if g_env.rank == dst:
            t[needed_rows_idx] = p2p_buf
            # g_logger.log('p2p dst done', src, dst)
            return


def broad_func(node_count, am_partitions, inputs):
    device = g_env.device
    n_per_proc = math.ceil(float(node_count) / g_env.world_size)
    z_loc = torch.zeros((am_partitions[0].size(0), inputs.size(1)), device=device)
    inputs_recv = torch.zeros((n_per_proc, inputs.size(1)), device=device)

    for i in range(g_env.world_size):
        if i == g_env.rank:
            inputs_recv = inputs.clone()
        elif i == g_env.world_size - 1:
            inputs_recv = torch.zeros((am_partitions[i].size(1), inputs.size(1)), device=device)

        g_timer.barrier_all()
        torch.cuda.synchronize()

        g_timer.start('broadcast')
        # dist.broadcast(inputs_recv, src=i, group=g_env.world_group)
        p2p_broadcast(inputs_recv, i)
        torch.cuda.synchronize()  # comm or comp?
        g_timer.stop('broadcast')#,'comm')

        g_timer.barrier_all()
        torch.cuda.synchronize()

        g_timer.start('spmm')
        spmm_gpu(am_partitions[i].indices()[0].int(), am_partitions[i].indices()[1].int(), 
                        am_partitions[i].values(), am_partitions[i].size(0), 
                        am_partitions[i].size(1), inputs_recv, z_loc)

        torch.cuda.synchronize()
        g_timer.stop('spmm')#, 'comp')
        g_timer.barrier_all()
    return z_loc


class GCNFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adj_matrix, am_partitions,  activation_func):
        ctx.save_for_backward(inputs, weight, adj_matrix)
        ctx.am_partitions = am_partitions
        ctx.activation_func = activation_func
        z = broad_func(adj_matrix.size(0), am_partitions, inputs)

        torch.cuda.synchronize()
        g_timer.start('mm')
        z = torch.mm(z, weight)
        torch.cuda.synchronize()
        g_timer.stop('mm') #, 'comp')

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

        torch.cuda.synchronize()
        g_timer.start('mm')
        grad_input = torch.mm(ag, weight.t())
        torch.cuda.synchronize()
        g_timer.stop('mm')#, 'comp')

        # Second backprop equation (reuses the A * G^l computation)
        grad_weight = outer_product2(inputs.t(), ag)

        return grad_input, grad_weight, None, None, None, None, None, None


def train(inputs, weight1, weight2, adj_matrix, am_partitions, optimizer, local_train_mask, local_labels):
    outputs = GCNFunc.apply(inputs, weight1, adj_matrix, am_partitions, F.relu)
    outputs = GCNFunc.apply(outputs, weight2, adj_matrix, am_partitions, lambda x:F.log_softmax(x, dim=1))

    optimizer.zero_grad()

    if list(local_labels[local_train_mask].size())[0] > 0:
        loss = F.nll_loss(outputs[local_train_mask], local_labels[local_train_mask])
        loss.backward()
    else:
        fake_loss = (outputs * torch.cuda.FloatTensor(outputs.size(), device=device).fill_(0)).sum()
        fake_loss.backward()

    optimizer.step()
    return outputs


def test(outputs, vertex_count):
    logits, accs = outputs, []
    for mask in [g_data.g.train_mask, g_data.g.val_mask, g_data.g.test_mask]:
        pred = logits[mask].max(1)[1]
        acc = pred.eq(g_data.g.labels[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


def main():
    global run
    inputs_loc, adj_matrix_loc, am_pbyp = g_data.local_features, g_data.local_adj, g_data.local_adj_parts 
    device = g_env.device

    torch.cuda.synchronize()

    for i in range(args.run_count):
        run = i
        torch.manual_seed(0)
        weight1_nonleaf = torch.rand(g_data.g.num_features, args.mid_layer, requires_grad=True, device=device)
        weight1_nonleaf.retain_grad()

        weight2_nonleaf = torch.rand(args.mid_layer, g_data.g.num_classes, requires_grad=True, device=device)
        weight2_nonleaf.retain_grad()

        weight1 = Parameter(weight1_nonleaf)
        weight2 = Parameter(weight2_nonleaf)

        optimizer = torch.optim.Adam([weight1, weight2], lr=0.01)

        local_train_mask = torch.split(g_data.g.train_mask.bool(), am_pbyp[0].size(0), dim=0)[g_env.rank]
        local_labels = torch.split(g_data.g.labels, am_pbyp[0].size(0), dim=0)[g_env.rank]

        for epoch in range(args.epochs):
            g_timer.start('train')
            outputs = train(inputs_loc, weight1, weight2, adj_matrix_loc, am_pbyp, optimizer, local_train_mask, local_labels)
            g_timer.stop('train')
            if epoch%10==0:
                g_logger.log("Epoch: {:03d}".format(epoch))

    n_per_proc = math.ceil(g_data.g.features.size(0) / g_env.world_size)
    output_parts = [torch.zeros(n_per_proc, g_data.g.num_classes, device=g_env.device) for i in range(g_env.world_size)]

    if outputs.size(0) != n_per_proc:
        pad_row = n_per_proc - outputs.size(0) 
        outputs = torch.cat((outputs, torch.cuda.FloatTensor(pad_row, g_data.g.num_classes, device=g_env.device)), dim=0)

    dist.all_gather(output_parts, outputs)
    output_parts[g_env.rank] = outputs
    
    padding = g_data.g.features.size(0) - n_per_proc * (g_env.world_size - 1)
    output_parts[g_env.world_size - 1] = output_parts[g_env.world_size - 1][:padding,:]
    outputs = torch.cat(output_parts, dim=0)

    train_acc, val_acc, test_acc = test(outputs, am_pbyp[0].size(1))
    g_logger.log(g_timer.summary())
    g_logger.log( 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(args.epochs, train_acc, val_acc, test_acc))
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run_count", type=int, default=1)
    parser.add_argument("--graphname", type=str, default="SmallerReddit")
    parser.add_argument("--timing", type=bool, default=True)
    parser.add_argument("--mid_layer", type=int, default=16)
    args = parser.parse_args()
    print(args)
    g_env = utils.DistEnv(args.local_rank, args.world_size, args.backend)

    g_timer = utils.DistTimer(g_env)
    g_logger = utils.DistLogger(g_env)
    g_logger.log('dist env inited:', g_env.backend, g_env.world_size)

    g_data = DistData(g_env, args.graphname)
    g_logger.log('dist data inited', args.graphname)

    main()

