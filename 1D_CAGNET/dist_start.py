import os
import torch
import argparse
import datetime as dt
import torch.distributed as dist
from torch.multiprocessing import Process

import gcn_distr
import dist_data_util


# distributed on single-machine multi-GPU


def dist_log(*args, rank,  group=None):
    assert(rank>=0)
    if group is None:
        print(dt.datetime.now(), '[Rank %2d] '%rank, end='')
        print(*args)
    else:
        assert(group is not None)
        dist.barrier(group)
        if rank==0:
            print(dt.datetime.now(), '[Rank all] ', end='')
            print(*args)


#def dist_bootstrap(rank, dist_main, A_blocks, A_block_seps, H_blocks, labels, num_classe, args):  # called by torch with rank
def dist_bootstrap(rank, dist_main, A, H, labels, num_classes, args):  # called by torch with rank
    world_size = args.nprocs
    if args.backend=='nccl':
        device = torch.device('cuda', rank+4)
    else:
        device = torch.device('cpu')
    dist_log(str(device), rank=rank)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # os.environ['NCCL_DEBUG']='INFO'
    os.environ['NCCL_SOCKET_IFNAME']='lo'
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)

    p2p_group_dict = {}
    world_group = dist.new_group(list(range(world_size)))
    for src in range(world_size):
        for dst in range(src+1, world_size):
            p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
            p2p_group_dict[(dst, src)] = p2p_group_dict[(src, dst)]
    dist_log('P2P groups inited', rank=rank, group=world_group)

    dist_main(device, rank, world_size, world_group, p2p_group_dict, A, H, labels, num_classes, args)


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
    print('data loaded')
    # A_blocks, A_block_seps, H_blocks = dist_data_util.partition_1D(A, H, args.nprocs)

    #dist_boot_args=(gcn_distr.dist_main, A_blocks, A_block_seps, H_blocks, labels, num_classes, args)
    dist_boot_args=(gcn_distr.dist_main, A, H, labels, num_classes, args)
    torch.multiprocessing.spawn(dist_bootstrap, dist_boot_args, args.nprocs)  # do not killing 1 by 1.


if __name__ == "__main__":
    begin = dt.datetime.now()
    print(__file__, 'begin')

    main()

    end = dt.datetime.now()
    print('total time cost:', end-begin)
