from dist_1d import *  # global vars


def dist_log(*args, group=None):
    assert(g_rank>=0)
    if group is None:
        print(dt.datetime.now(), '[Rank %2d] '%g_rank, end='')
        print(*args, flush=True)
        with open('all_log_%d.txt'%g_rank, 'a+') as f:
            print(dt.datetime.now(), '[Rank %2d] '%g_rank, end='', file=f)
            print(*args, file=f, flush=True)
    else:
        assert(group is not None)
        dist.barrier(group)
        if g_rank==rank:
            print(dt.datetime.now(), '[Rank all] ', end='')
            print(*args)


