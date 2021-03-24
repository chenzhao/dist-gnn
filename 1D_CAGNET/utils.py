
class DistBase():
    def __init__(self, local_rank, world_size, world_group):
        assert(rank>=0)
        self.rank = local_rank
        self.world_size = world_size
        self.group = world_group


class DistUtil(DistBase):
    def log(self, *args):
        head = '%s [Rank %2d] '%(dt.datetime.now(), self.rank)
        print(head, *args, flush=True)
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


class DistData(DistBase):
    @static
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

    def oned_partition(self, inputs, adj_matrix, adj_values, data, features, classes):
        node_count = inputs.size(0)
        n_per_proc = math.ceil(float(node_count) / self.world_size)
        am_partitions = None
        am_pbyp = None

        with torch.no_grad():
            am_partitions, av_partitions, vtx_indices = split_coo_with_values(adj_matrix, adj_values, node_count, n_per_proc, 1)

            proc_node_count = vtx_indices[self.rank + 1] - vtx_indices[self.rank]
            am_pbyp, av_pbyp, _ = split_coo_with_values(am_partitions[self.rank], av_partitions[self.rank], node_count, n_per_proc, 0)
            for i in range(len(am_pbyp)):
                uni_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], av_pbyp[i], size=(uni_node_count, proc_node_count), requires_grad=False).coalesce()
            for i in range(len(am_partitions)):
                proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
                am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], av_partitions[i], size=(node_count, proc_node_count), requires_grad=False).coalesce()
            input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / self.world_size), dim=0)
        print('Rank',self.rank,'parted')
        return  input_partitions[self.rank], am_partitions[self.rank], am_pbyp



