import os
import torch
import datetime as dt
import torch.distributed as dist
from torch.multiprocessing import Process

import gcn_distr
import dist_data_util


# distributed on single-machine multi-GPU

g_rank = -1
g_world_size = -1
g_world_group = None
g_p2p_group_dict = {}
device = None


def dist_log(*args, sep=True):
    assert(g_rank>=0)
    assert(g_world_group is not None)
    if sep:
        print(dt.datetime.now(), '[Rank %2d] '%g_rank, end='')
        print(*args)
    else:
        dist.barrier(g_world_group)
        if g_rank==0:
            print(dt.datetime.now(), '[Rank all] ', end='')
            print(*args)


def dist_bootstrap(rank, dist_main, A_blocks, A_block_seps, H_blocks, labels, num_classe, args):  # called by torch with rank
    global g_world_size , g_rank , device
    g_world_size = nprocs
    g_rank = rank
    if args.backend=='nccl':
        device = torch.device('cuda', rank)
    else:
        device = torch.device('cpu')

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # os.environ['NCCL_DEBUG']='INFO'
    os.environ['NCCL_SOCKET_IFNAME']='lo'
    dist.init_process_group(backend, rank=rank, world_size=g_world_size)

    global g_world_group, g_p2p_group_dict
    g_world_group = dist.new_group(list(range(g_world_size)))
    for src in range(g_world_size):
        for dst in range(src+1, g_world_size):
            g_p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
            g_p2p_group_dict[(dst, src)] = g_p2p_group_dict[(src, dst)]
    dist_log('P2P groups inited', each=False)

    dist_main(g_rank, g_world_size, A_blocks, A_block_seps, H_blocks, labels, num_classes, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nprocs", type=int, default=8)
    parser.add_argument("--backend", type=str, default="nccl")  # or gloo
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--graphname", type=str, default="Reddit")
    parser.add_argument("--timing", type=bool, default=True)
    parser.add_argument("--mid_layer", type=int, default=16)
    parser.add_argument("--normalization", type=bool, default=True)
    parser.add_argument("--accuracy", type=bool, default=True)
    parser.add_argument("--download", type=bool, default=False)
    args = parser.parse_args()
    print(args)

    A, H, labels, num_classes = dist_data_util.load_data(args) # coo_adj_matrix, features
    A_blocks, A_block_seps, H_blocks = dist_data_util.partition_1D(A, H, args.nprocs)

    dist_boot_args=(gcn_distr.dist_main, A_blocks, A_block_seps, H_blocks, labels, num_classes, args)
    torch.multiprocessing.spawn(dist_bootstrap, dist_boot_args, args.nprocs)  # do not killing 1 by 1.


if __name__ == "__main__":
    begin = dt.datetime.now()
    print(__file__, 'begin')

    main()

    end = dt.datetime.now()
    print('total time cost:', end-begin)
