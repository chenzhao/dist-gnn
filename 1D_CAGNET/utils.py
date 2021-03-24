import datetime as dt
import torch
import torch.distributed as dist
from fast_reddit import Reddit, SmallerReddit
import math

class DistEnv():
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


class DistUtil():
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
        self.local_features, self.local_adj, self.local_adj_parts = \
            DistData.CAGNET_oned(self.rank, self.world_size, self.features, self.adj_indices, self.adj_values, self.features)
        self.local_features = self.local_features.to(self.device)
        self.local_adj = self.local_adj.to(self.device)
        for i in range(self.world_size):
            self.local_adj_parts[i] = self.local_adj_parts[i].t().coalesce().to(self.device)

    @staticmethod
    def split_coo_with_values(adj_matrix, adj_values, node_count, n_per_proc, dim):
        vtx_indices = list(range(0, node_count, n_per_proc))
        vtx_indices.append(node_count)

        am_partitions = []
        av_partitions = []
        for i in range(len(vtx_indices) - 1):
            mask = ((adj_matrix[dim,:] >= vtx_indices[i]) & (adj_matrix[dim,:] < vtx_indices[i + 1])).nonzero().squeeze(1)
            am_part = adj_matrix[:,mask]
            am_part[dim] -= vtx_indices[i]
            am_partitions.append(am_part)

            av_part = adj_values[mask]
            av_partitions.append(av_part)

        return am_partitions, av_partitions, vtx_indices

    @staticmethod
    def CAGNET_oned(rank, world_size, inputs, adj_matrix, adj_values, features):
        node_count = inputs.size(0)
        n_per_proc = math.ceil(node_count/world_size)
        with torch.no_grad():
            am_partitions, av_partitions, vtx_indices = DistData.split_coo_with_values(adj_matrix, adj_values, node_count, n_per_proc, 1)

            proc_node_count = vtx_indices[rank + 1] - vtx_indices[rank]
            am_pbyp, av_pbyp, _ = DistData.split_coo_with_values(am_partitions[rank], av_partitions[rank], node_count, n_per_proc, 0)
            for i in range(len(am_pbyp)):
                uni_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], av_pbyp[i], size=(uni_node_count, proc_node_count), requires_grad=False).coalesce()
            for i in range(len(am_partitions)):
                proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], av_partitions[i], size=(node_count, proc_node_count), requires_grad=False).coalesce()
            input_partitions = torch.split(inputs, math.ceil(inputs.size(0)/world_size), dim=0)
        # print('Rank',rank,'parted')
        return  input_partitions[rank], am_partitions[rank], am_pbyp



