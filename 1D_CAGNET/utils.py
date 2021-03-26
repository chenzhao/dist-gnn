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


class DistData(DistUtil):
    def __init__(self, env, graph_name):
        super().__init__(env)
        if graph_name == "Reddit":
            data = Reddit()
        elif graph_name == "SmallerReddit":
            data = SmallerReddit()
        else:
            assert False
        # print('data loaded to host mem')
        self.features = data.x
        self.labels = data.y.to(self.device)
        self.adj_indices = data.edge_index
        self.adj_values = data.normalized_adj_values
        self.num_features = data.x.size(1)
        self.num_classes = torch.unique(data.y).size(0)
        self.train_mask, self.val_mask, self.test_mask = data.train_mask, data.val_mask, data.test_mask
        self.partition_1d()
        # print('Rank',self.rank,'data ready')

    def partition_1d(self):
        self.local_features, self.local_adj, self.local_adj_parts, self.nz_col_dict = \
            DistData.CAGNET_oned(self.rank, self.world_size, self.features, self.adj_indices, self.adj_values)
        self.local_features = self.local_features.to(self.device)
        self.local_adj = self.local_adj.to(self.device)
        for i in range(self.world_size):
            self.local_adj_parts[i] = self.local_adj_parts[i].t().coalesce().to(self.device)

    @staticmethod
    def split_coo_with_values(adj_matrix, adj_values, node_count, world_size, dim):
        n_per_proc = math.ceil(node_count/world_size)
        sep = list(range(0, node_count, n_per_proc))
        sep.append(node_count)

        am_partitions = []
        av_partitions = []
        for i in range(world_size):
            mask = ((adj_matrix[dim,:] >= sep[i]) & (adj_matrix[dim,:] < sep[i+1])).nonzero().squeeze(1)
            am_part = adj_matrix[:,mask]
            am_part[dim] -= sep[i]
            am_partitions.append(am_part)
            av_partitions.append(adj_values[mask])

        return am_partitions, av_partitions, sep

    @staticmethod
    def CAGNET_oned(rank, world_size, inputs, adj_indices, adj_values):
        node_count = inputs.size(0)
        nz_col_dict = {}
        with torch.no_grad():
            am_partitions, av_partitions, sep = DistData.split_coo_with_values(adj_indices, adj_values, node_count, world_size, 1)
            sizes = [sep[i+1]-sep[i] for i in range(world_size)]
            am_pbyp, av_pbyp, _ = DistData.split_coo_with_values(am_partitions[rank], av_partitions[rank], node_count, world_size, 0)
            for i in range(world_size):
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], av_pbyp[i], size=(sizes[i], sizes[rank]), requires_grad=False).coalesce()
                # find nz cols for p2p bcast
                for j in range(len(am_pbyp)):
                    if i!=j:
                        i_mask = ((adj_indices[0,:]>=sep[i])&(adj_indices[0,:]<sep[i+1]))
                        j_mask = ((adj_indices[1,:]>=sep[j])&(adj_indices[1,:]<sep[j+1]))
                        col_ij = adj_indices[1, (i_mask & j_mask).nonzero().squeeze(1)] - sep[j]
                        nz_col_dict[(i,j)] = torch.unique(col_ij)
                        if rank==0:
                            print('nz col',i,j, nz_col_dict[(i,j)].size() )
            for i in range(world_size):
                am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], av_partitions[i], size=(node_count, sizes[i]), requires_grad=False).coalesce()
            input_partitions = torch.split(inputs, math.ceil(inputs.size(0)/world_size), dim=0)
        # print('Rank',rank,'parted')
        return input_partitions[rank], am_partitions[rank], am_pbyp, nz_col_dict



