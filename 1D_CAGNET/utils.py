import os
import datetime as dt
import torch
import torch.distributed as dist
from fast_reddit import Reddit, SmallerReddit
import math


class DistEnv:
    def __init__(self, local_rank, world_size):
        assert(local_rank>=0)
        assert(world_size>0)
        self.rank, self.world_size  = local_rank, world_size
        self.init_device()
        self.init_dist_groups()

    def init_device(self):
        self.local_device = torch.device('cuda', self.rank)
        torch.cuda.set_device(self.local_device)

    def barrier_all(self):
        dist.barrier(self.world_group)

    def init_dist_groups(self):
        dist.init_process_group(backend='nccl')
        self.world_group = dist.new_group(list(range(self.world_size)))
        self.p2p_group_dict = {}
        for src in range(self.world_size):
            for dst in range(src+1, self.world_size):
                self.p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
                self.p2p_group_dict[(dst, src)] = self.p2p_group_dict[(src, dst)]
        # print('dist groups inited')


class DistUtil:
    def __init__(self, env):
        self.env = env
        self.rank = env.rank
        self.device = env.local_device
        self.world_size = env.world_size
        self.world_group = env.world_group


class DistLogger(DistUtil):
    def log(self, *args):
        head = '%s [%1d] '%(dt.datetime.now().time(), self.rank)
        print(head+' '.join(map(str, args))+'\n', end='', flush=True)  # to prevent line breaking
        with open('all_log_%d.txt'%self.rank, 'a+') as f:
            print(head, *args, file=f, flush=True)

    def log_one(self, *args, rank=-1, group=None):
        if group is None:
            print(dt.datetime.now(), '[Rank %2d] '%self.rank, end='')
            print(*args, flush=True)
            with open('all_log_%d.txt'%g_rank, 'a+') as f:
                print(dt.datetime.now(), '[Rank %2d] '%self.rank, end='', file=f)
                print(*args, file=f, flush=True)
        else:
            assert(group is not None)
            dist.barrier(group)
            if self.rank==rank:
                print(dt.datetime.now(), '[Rank all] ', end='')
                print(*args)



def some_value():
    print('some value')
    return 1233

class O:
    p = some_value()
    def __init__(self):
        self.p+=1
        print('O init', self.p)

class OO(O):
    p=233


if __name__ == '__main__':
    o = O()
    o2 = OO()
    o3 = O()
