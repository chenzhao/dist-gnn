import os
import datetime as dt
import math
import torch
import torch.distributed as dist
from torch.multiprocessing import Process


g_rank = -1

g_world_size = -1
g_world_group = None
g_p2p_group_dict = {}
device = None

def dist_log(*args):
    print(dt.datetime.now(), '[Rank %2d] '%g_rank, end='')
    print(*args)


def test_p2p_on_broadcast(src=0, dst=1):
    # tsize = (1024,1024)
    tsize = (2, 10,1024,1024)
    # nccl 1024^3 12s 2*1024^3 24s
    # gloo 1024^3 12s 2*1024^3 24s  nearly the same as nccl
    if g_rank==src:
        ts = torch.ones(tsize, dtype=torch.uint8).to(device)
    elif g_rank==dst:
        ts = torch.zeros(tsize, dtype=torch.uint8).to(device)
    else:
        return
    # dist_log('broadcast begin', g_rank, device, 'src', src, 'dst', dst)
    # dist_log('broadcast init bytes:', math.prod(tsize))
    group = g_p2p_group_dict[(src, dst)]
    begin = dt.datetime.now()
    if g_rank==src:
        #dist_log('src begin broadcast to', dst)
        dist.broadcast(tensor=ts, src=src, group=group)
        # dist.send(ts, dst)
        end = dt.datetime.now()
        dist_log('src subgroup done ', src, dst,' time', end-begin, ts.view(-1)[0])
    else:
        # dist_log('dst begin broadcast recv from', src)
        dist.broadcast(tensor=ts, src=src, group=group)
        # dist.recv(ts,  src)
        end = dt.datetime.now()
        dist_log('dst subgroup done', src, dst,' time', end-begin, ts.view(-1)[0])
    dist.barrier(group)


def test_broadcast(src=0, group=None):
    # nccl 1024^3 23s 2*1024^3 45s
    # gloo 1024^3 xxs 2*1024^3 1m14s
    tsize = (2, 10,1024,1024)
    if g_rank==src:
        ts = torch.ones(tsize, dtype=torch.uint8).to(device)
    else:
        ts = torch.zeros(tsize, dtype=torch.uint8).to(device)
    # dist_log('full broadcast init bytes:', math.prod(tsize))
    begin = dt.datetime.now()
    if not group:
        dist.broadcast(tensor=ts, src=src)
    else:
        dist.broadcast(tensor=ts, src=src, group=group)
    end = dt.datetime.now()
    dist_log('full broadcast cost', end-begin, ts.view(-1)[0])
    dist.barrier()


def test_func(rank):
    # dist_log('process inited:', rank,'of', g_world_size)
    # test_broadcast()
    # group = dist.new_group([0,1,2,3])
    begin = dt.datetime.now()
    for i in range(1000):
        for i in range(1):
            test_broadcast(0)
            # test_p2p_on_broadcast(0, 1)
            # test_p2p_on_broadcast(2, 3)
            # test_p2p_on_broadcast(4, 5)
            # test_p2p_on_broadcast(6, 7)
            # test_p2p_on_broadcast(2, 3)
            # test_p2p_on_broadcast(i, i+1)
        # test_p2p_on_broadcast(1, 2)
        # test_p2p_on_broadcast(2, 3)
        # test_broadcast(0)
        # test_broadcast(1)
        # test_broadcast(2)
    end = dt.datetime.now()

    dist.barrier()
    dist_log('total time: ', end-begin)
    return

    tsize = (1024,1024,0)
    if rank%2==0:
        ts = torch.ones(tsize, dtype=torch.uint8).to(device) * rank
        dist_log('init bytes:', math.prod(tsize))
        dist.send(tensor=ts, dst=(rank+1)%g_world_size)
        dist_log('sent', ts.view(-1)[0])
    else:
        tr = torch.ones(tsize, dtype=torch.uint8).to(device)
        dist.recv(tensor=tr, src=(rank-1)%g_world_size)
        dist_log('recvd', tr.view(-1)[0])


def dist_process_main(rank, nprocs, func, backend):
    global g_world_size , g_rank , device
    g_world_size = nprocs
    g_rank = rank
    if backend=='nccl':
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
            # print('init group', src, dst)
            g_p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
            g_p2p_group_dict[(dst, src)] = g_p2p_group_dict[(src, dst)] 

    func(rank)


def main(nprocs=8):
    backend = 'nccl'
    # backend = 'gloo'
    args=(nprocs, test_func, backend)
    torch.multiprocessing.spawn(dist_process_main, args, nprocs)


if __name__ == "__main__":
    begin = dt.datetime.now()
    dist_log('main begin')
    
    main()

    end = dt.datetime.now()
    dist_log('total time cost:', end-begin)

