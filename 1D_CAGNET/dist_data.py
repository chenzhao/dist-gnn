import os
import math
import time
import torch
import numpy as np
import os.path as osp
from collections import defaultdict


from utils import DistUtil
from coo_graph import COO_Graph


class DistData(DistUtil):
    def __init__(self, env, graph_name):
        super().__init__(env)
        self.graph_name = graph_name
        self.g = COO_Graph(graph_name)
        # print('data loaded to host mem')
        if os.path.exists(self.parted_data_file()):# and False:
            self.load_local_part()
        else:
            self.local_features, self.local_adj, self.local_adj_parts, self.nz_col_dict = \
                DistData.CAGNET_oned(self.rank, self.world_size, self.g.features, self.g.DAD_idx, self.g.DAD_val)
            self.save_local_part()
        # print('Rank',self.rank,'data ready')
        self.data_to_device()

    def load_local_part(self):
        d = torch.load(self.parted_data_file())
        to_sparse = lambda tpl: torch.sparse_coo_tensor(*tpl)
        self.local_features = d['feat']
        self.local_adj = to_sparse(d['adj'])
        self.local_adj_parts = [to_sparse(tpl) for tpl in d['parts']]
        self.nz_col_dict = d['nz']
        pass

    def save_local_part(self):
        sparse_tuple = lambda t: (t._indices(), t._values(),  t.size())
        os.makedirs(os.path.dirname(self.parted_data_file()), exist_ok=True)
        f = self.local_features.clone()
        a = self.local_adj.clone()
        pp = [t.clone() for t in self.local_adj_parts]
        nz = self.nz_col_dict
        d = {'feat':f, 'adj':sparse_tuple(a), 'parts':[sparse_tuple(p) for p in pp], 'nz': nz}

        #print('save local')
        #print('g:', self.g.num_nodes, self.g.num_edges, self.g.num_features, self.g.num_classes)
        #print(self.local_features.size())
        #print(self.local_adj.size())
        #print(self.local_adj_parts[0].size())
        torch.save(d, self.parted_data_file())

    def parted_data_file(self):
        return os.path.join('..', 'coo_graph_data', self.graph_name, 'parted', 'part%d_of_%d.pt'%(self.rank, self.world_size))

    def data_to_device(self):
        self.local_features = self.local_features.to(self.device)
        self.local_adj = self.local_adj.to(self.device)
        for i in range(self.world_size):
            self.local_adj_parts[i] = self.local_adj_parts[i].t().coalesce().to(self.device)
        self.g.labels = self.g.labels.float().to(self.device)

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
        # print(rank, 'total nz', adj_values.size())
        am_partitions, av_partitions, sep = DistData.split_coo_with_values(adj_indices, adj_values, node_count, world_size, 1)
        sizes = [sep[i+1]-sep[i] for i in range(world_size)]
        am_pbyp, av_pbyp, _ = DistData.split_coo_with_values(am_partitions[rank], av_partitions[rank], node_count, world_size, 0)
        # print('all nz', rank, av_partitions[rank].size())
        for i in range(world_size):
            print('block nz', rank, i, av_pbyp[i].size())
            am_pbyp[i] = torch.sparse_coo_tensor(am_pbyp[i], av_pbyp[i], size=(sizes[i], sizes[rank])).coalesce()
            # find nz cols for p2p bcast
            for j in range(len(am_pbyp)):
                if i!=j:
                    i_mask = ((adj_indices[0,:]>=sep[i])&(adj_indices[0,:]<sep[i+1]))
                    j_mask = ((adj_indices[1,:]>=sep[j])&(adj_indices[1,:]<sep[j+1]))
                    col_ij = adj_indices[1, (i_mask & j_mask).nonzero().squeeze(1)] - sep[j]
                    nz_col_dict[(i,j)] = torch.unique(col_ij)
                    if rank==0:
                        # print('nz col',i,j, nz_col_dict[(i,j)].size() )
                        pass
        am_local = torch.sparse_coo_tensor(am_partitions[rank], av_partitions[rank], size=(node_count, sizes[rank])).coalesce()
        input_partitions = torch.split(inputs, math.ceil(inputs.size(0)/world_size), dim=0)
        # print('Rank',rank,'parted')
        return input_partitions[rank], am_local, am_pbyp, nz_col_dict


if __name__ == '__main__':
    pass
