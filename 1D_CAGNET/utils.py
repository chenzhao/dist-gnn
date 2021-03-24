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

def oned_partition(inputs, adj_matrix, adj_values, data, features, classes):
    node_count = inputs.size(0)
    n_per_proc = math.ceil(float(node_count) / g_world_size)
    am_partitions = None
    am_pbyp = None

    with torch.no_grad():
        am_partitions, av_partitions, vtx_indices = split_coo_with_values(adj_matrix, adj_values, node_count, n_per_proc, 1)

        proc_node_count = vtx_indices[g_rank + 1] - vtx_indices[g_rank]
        am_pbyp, av_pbyp, _ = split_coo_with_values(am_partitions[g_rank], av_partitions[g_rank], node_count, n_per_proc, 0)
        for i in range(len(am_pbyp)):
            uni_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], av_pbyp[i], size=(uni_node_count, proc_node_count), requires_grad=False).coalesce()
        for i in range(len(am_partitions)):
            proc_node_count = vtx_indices[i + 1] - vtx_indices[i]
            am_partitions[i] = torch.sparse_coo_tensor(am_partitions[i], av_partitions[i], size=(node_count, proc_node_count), requires_grad=False).coalesce()
        input_partitions = torch.split(inputs, math.ceil(float(inputs.size(0)) / g_world_size), dim=0)
    print('Rank',g_rank,'parted')
    return  input_partitions[g_rank], am_partitions[g_rank], am_pbyp



