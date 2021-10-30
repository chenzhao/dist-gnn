import os
import datetime as dt
import torch
import torch.distributed as dist
import math
import time
import pickle
import statistics
from collections import defaultdict


class DistEnv:
    def __init__(self, local_rank, world_size, rep=1, backend='nccl'):
        assert(local_rank>=0)
        assert(world_size>0)
        self.rank, self.world_size = local_rank, world_size
        self.rep = rep
        self.backend = backend
        self.init_device()
        self.init_dist_groups()
        if self.rep>1:
            self.init_rep_groups()
            self.rank_grp = self.rank // self.rep
            self.rank_col = self.rank % self.rep
            self.total_grp = self.rank // self.world_size
            if self.rank_grp >= self.total_grp:
                raise Exception('invalid args')

    def init_device(self):
        self.device = torch.device('cuda', self.rank)
        torch.cuda.set_device(self.device)

    def barrier_all(self):
        dist.barrier(self.world_group)

    def init_rep_groups(self):
        self.row_procs = []
        for i in range(0, self.world_size, self.rep):
            self.row_procs.append(list(range(i, i + self.rep)))
        self.col_procs = []
        for i in range(self.rep):
            self.col_procs.append(list(range(i, self.world_size, self.rep)))
        row_groups = []
        for i in range(len(self.row_procs)):
            self.row_groups.append(dist.new_group(self.row_procs[i]))
        self.col_groups = []
        for i in range(len(self.col_procs)):
            self.col_groups.append(dist.new_group(self.col_procs[i]))
        return self.row_groups, self.col_groups

    def init_dist_groups(self):
        dist.init_process_group(backend=self.backend)
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
        self.store = dist.FileStore("/tmp/torch_filestore", self.world_size)


class DistLogger(DistUtil):
    def log(self, *args, oneline=False, rank=-1):
        if rank!=-1 and self.rank!=rank:
            return
        head = '%s [%1d] '%(dt.datetime.now().time(), self.rank)
        tail = '\r' if oneline else '\n'
        the_whole_line = head+' '.join(map(str, args))+tail
        print(the_whole_line, end='', flush=True)  # to prevent line breaking
        with open('all_log_%d.txt'%self.rank, 'a+') as f:
            print(the_whole_line, end='', file=f, flush=True)  # to prevent line breaking

class DistTimer(DistUtil):
    def __init__(self, env):
        super().__init__(env)
        self.start_time_dict = {}
        self.duration_dict = defaultdict(float)
        self.count_dict = defaultdict(int)

    def summary(self):
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %5d %s" % (self.duration_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def sync_duration_dicts(self):
        self.store.set('duration_dict_%d'%self.rank, pickle.dumps(self.duration_dict))
        self.env.barrier_all()
        self.all_durations = [pickle.loads(self.store.get('duration_dict_%d'%rank)) for rank in range(self.world_size)]

    def summary_all(self):
        self.sync_duration_dicts()
        avg_dict = {}
        std_dict = {}
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            avg_dict[key], std_dict[key] = statistics.mean(data), statistics.stdev(data)
        s = '\ntimer summary:\n' +  "\n".join("%6.2fs %6.2fs %5d %s" % (avg_dict[key], std_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return s

    def barrier_all(self):
        return
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
        log(key)
        for subkey in other_keys:
            log(key+'-'+subkey)
        return


if __name__ == '__main__':
    pass

