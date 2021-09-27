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
    def __init__(self, local_rank, world_size, backend='nccl'):
        assert(local_rank >= 0)
        assert(world_size > 0)
        self.rank, self.world_size = local_rank, world_size
        self.backend = backend
        self.init_device()
        self.init_dist_groups()

    def init_device(self):
        self.device = torch.device('cuda', self.rank)
        torch.cuda.set_device(self.device)

    def barrier_all(self):
        dist.barrier(self.world_group)

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
    def __init__(self, env: DistEnv):
        self.env = env
        self.rank = env.rank
        self.device = env.device
        self.world_size = env.world_size
        self.world_group = env.world_group
        self.store = dist.FileStore("/tmp/torch_filestore", self.world_size)


class DistLogger(DistUtil):
    def __init__(self, env: DistEnv):
        super().__init__(env)
        self.logs = []

    def log(self, line, rank=-1):
        if rank != -1 and rank != self.rank:
            return
        whole_line = f'{dt.datetime.now().time()} [{self.rank}] {line}'
        # print(whole_line, end='', flush=True)  # to prevent line breaking
        self.logs.append(whole_line)

    def sync_duration_dicts(self):
        self.store.set(f'dist_log_{self.rank}', pickle.dumps(self.logs))
        self.env.barrier_all()
        return [pickle.loads(self.store.get(f'dist_log_{rank}')) for rank in range(self.world_size)]

    def summary_all(self):
        self.sync_duration_dicts()
        return '\n'.join(self.logs)


class DistTimer(DistUtil):
    def __init__(self, env: DistEnv):
        super().__init__(env)
        self.start_time_dict = {}
        self.duration_array = []

    def summary(self):
        s = '\ntimer summary:\n' + "\n".join("%6.2fs %s" % (item['duration'], item['key']) for item in self.duration_array)
        return s

    def sync_duration_dicts(self):
        self.store.set('duration_array_%d' % self.rank, pickle.dumps(self.duration_array))
        self.env.barrier_all()
        return [pickle.loads(self.store.get('duration_array_%d' % rank)) for rank in range(self.world_size)]

    def summary_all(self):
        all_durations = self.sync_duration_dicts()
        avg_dict = {}
        std_dict = {}
        for idx in range(len(self.duration_array)):
            key = self.duration_array[idx]['key']
            data = [d[idx]['duration'] for d in all_durations]
            avg_dict[key], std_dict[key] = statistics.mean(data), statistics.stdev(data)
        s = '\ntimer summary:\n' + "\n".join("%6.2fs %6.2fs %s" % (avg_dict[key], std_dict[key], key) for key in avg_dict)
        return s

    # def barrier_all(self):
    #     return
    #     self.start('barrier')
    #     self.env.barrier_all()
    #     self.stop('barrier')

    def start(self, key: str):
        self.start_time_dict[key] = time.time()
        return self.start_time_dict[key]

    def stop(self, key: str):
        d = time.time() - self.start_time_dict[key]
        self.duration_array.append({
            'key': key,
            'duration': d
        })
        # self.order_dict[key] = self.order
        # self.count_dict[key] += 1
        return


def mem_report(*tensors):
    '''
    Report the memory usage of the tensor.storage in pytorch
    There are two major storage types in our major concern:
        - GPU: tensors transferred to CUDA devices
        - CPU: tensors remaining on the system memory (usually unimportant)
    Args:
        - tensors: the tensors of specified type
        - mem_type: 'CPU' or 'GPU' in current implementation
    Returns:
        - total memory usage in MBytes
    '''
    # print('Storage on %s' %(mem_type))
    # print('-'*LEN)
    total_numel = 0
    total_mem = 0
    visited_data = []
    for tensor in tensors:
        if tensor.is_sparse:
            continue
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.append(data_ptr)

        numel = tensor.storage().size()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
        total_mem += mem
        # element_type = type(tensor).__name__
        # size = tuple(tensor.size())
        # print('%s\t\t%s\t\t%.2f' % (
        #     element_type,
        #     size,
        #     mem) )
    # print('-'*LEN)
    # print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
    # print('-'*LEN)
    return total_mem


# def my_broadcast(tensor, src, group)

if __name__ == '__main__':
    pass

