import os
import datetime as dt
import torch
import torch.distributed as dist
import math
import time
import statistics
from collections import defaultdict


class DistEnv:
    def __init__(self, local_rank, world_size):
        assert(local_rank>=0)
        assert(world_size>0)
        self.rank, self.world_size  = local_rank, world_size
        self.init_device()
        self.init_dist_groups()

    def init_device(self):
        self.device = torch.device('cuda', self.rank)
        torch.cuda.set_device(self.device)

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
        self.device = env.device
        self.world_size = env.world_size
        self.world_group = env.world_group


class DistLogger(DistUtil):
    def log(self, *args):
        head = '%s [%1d] '%(dt.datetime.now().time(), self.rank)
        print(head+' '.join(map(str, args))+'\n', end='', flush=True)  # to prevent line breaking
        with open('all_log_%d.txt'%self.rank, 'a+') as f:
            print(head, *args, file=f, flush=True)


class DistTimer(DistUtil):
    def __init__(self, env):
        super().__init__(env)
        self.start_time_dict = {}
        self.duration_dict = defaultdict(float)
        self.count_dict = defaultdict(int)

    def summary(self):
        s = "\n".join("%s %.4fs %4d" % (key, self.duration_dict[key], self.count_dict[key]) for key in self.duration_dict)
        return s

    def barrier_all(self, subset=False):
        self.start('barrier')
        self.env.barrier_all()
        self.stop('barrier')

    def start(self, key):
        self.start_time_dict[key] = time.time()
        return self.start_time_dict[key]

    def stop(self, key, *other_keys):
        def log(k, d=time.time() - self.start_time_dict[key]):
            self.duration_dict[k]+=d
            self.count_dict[k]+=1
        map(log, [key]+list(other_keys))
        return


if __name__ == '__main__':
    pass

