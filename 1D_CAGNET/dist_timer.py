import time
import statistics
from dist_1d import g_rank, g_world_group
from collections import defaultdict


class DistTimer():
    def __init__(rank,world_group):
        self.rank = rank
        self.world_group=world_group
        self.start_time_dict = defaultdict(float)
        self.duration_dict = defaultdict(float)
        pass

    def summary(self):
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
    def barrier_all(self, subset=False):
        barrier_tstart = time.time()
        dist.barrier(self.world_group)
        barrier_tstop = time.time()
        barrier_time[run][self.rank] += barrier_tstop - barrier_tstart
        if subset:
            barrier_subset_time[run][self.rank] += barrier_tstop - barrier_tstart


    def start_time(self, key):
        self.start_time_dict[key] = time.time()
        return self.start_time_dict[key]

    def stop_time(self, key, other_keys=None):
        assert ( key in self.start_time_dict )
        #barrier_all(g_world_group, g_rank)
        tstop = time.time()
        dur = tstop - self.start_time_dict[key]
        self.duration_dict[key] += dur
        if type(other_keys)==str:
            self.duration_dict[other_keys] += dur
        elif other_keys is not None:
            for k in other_keys:
                self.duration_dict[k] += dur
        return dur


