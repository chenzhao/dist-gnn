from dist_1d import g_rank, g_world_group


class DistTimer():
    def __init__(rank,world_group):
        self.rank = rank
        self.world_group=world_group
        pass

    def barrier_all(self, subset=False):
        barrier_tstart = time.time()
        dist.barrier(self.world_group)
        barrier_tstop = time.time()
        barrier_time[run][self.rank] += barrier_tstop - barrier_tstart
        if subset:
            barrier_subset_time[run][self.rank] += barrier_tstop - barrier_tstart


    def start_time(self, subset=False, src=None):
        # barrier_all(g_world_group, g_rank, subset)
        tstart = time.time()
        return tstart

    def stop_time(self, tstart):
        #barrier_all(g_world_group, g_rank)
        tstop = time.time()
        return tstop - tstart

