import os
import os.path
import numpy
import scipy.sparse
import torch
from download_torch_ready_data import Reddit, SmallerReddit
from torch_geometric.data import Data

from torch_geometric.utils.convert import to_networkx
import networkx
import metis



# a reorder example:
# parts        = 0,1,0,1,0,1
# node_ids     = 1,2,5,6,7,8
# will sort to = 1,5,7,2,6,8
# e.g. id=5 should be loc to idx=2(which is idx of old id=2), so change id=5 to 2.
# node_mapping = 1:1, 5:2, 7:5, ...

    # parted_ids = [(p,node_ids[old_node_idx]) for old_node_idx, p in enumerate(parts)]


def partition(r, P=8): # to be tensorified
    x, y, edge_index, split = map(r.data_dict.get, ['x', 'y', 'edge_index', 'split'])

    nxg = to_networkx(Data(edge_index=edge_index))
    print('partition begin')
    (edgecuts, parts) = metis.part_graph(nxg, P)
    print('parted', edgecuts, len(parts))
    node_ids = list(nxg)
    parted_ids = [(p,old_node_id) for p, old_node_id in zip(parts, node_ids)]
    sorted_node_ids = [node_id for p, node_id in sorted(parted_ids)]

    id_mapping = dict((old_id, new_id) for old_id, new_id in zip(sorted_node_ids, node_ids))
    old_edge_index = torch.clone(edge_index)

    edge_index[0] = torch.tensor([id_mapping[old_id.item()] for old_id in old_edge_index[0]])
    edge_index[1] = torch.tensor([id_mapping[old_id.item()] for old_id in old_edge_index[1]])
    print('node swapped')

    def swap_tensor(t):
        pass

    x = swap_tensor(x)
    y = swap_tensor(y)
    split = swap_tensor(split)



def main():
    # data_root = os.path.join('..', 'torch_ready_data', 'Reddit')
    # Reddit(data_root)
    data_root = os.path.join('..', 'torch_ready_data', 'SmallerReddit')
    r = SmallerReddit(data_root)

    # Cora()
    # Amazon()
    pass


if __name__ == '__main__':
    main()