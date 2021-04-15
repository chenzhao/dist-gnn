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


def metis_reorder(edge_index, x, y, split, P=8):
    # a reorder with gap example:
    # parts        = 0,1,0,1,0,1
    # node_ids     = 1,2,5,6,7,8
    # will sort to = 1,5,7,2,6,8
    # e.g. the old id=5 should be loc to idx=2(which is idx of old id=2), so change id=5 to 2.
    # e.g. id=7 should be loc to idx=3(which is idx of old id=5), so change id=7 to 5.
    # node_mapping = 1:1, 5:2, 7:5, ...

    nxg = to_networkx(Data(edge_index=edge_index))
    print('partition begin')
    (edgecuts, parts) = metis.part_graph(nxg, P)
    print('parted', edgecuts, len(parts))
    node_ids = list(nxg) # only works when no gap in ids
    parted_ids = [(p,old_node_id) for p, old_node_id in zip(parts, node_ids)]
    sorted_node_ids = [node_id for p, node_id in sorted(parted_ids)]

    id_mapping = dict((old_id, new_id) for old_id, new_id in zip(sorted_node_ids, node_ids))
    old_edge_index = torch.clone(edge_index)

    edge_index[0] = torch.tensor([id_mapping[old_id.item()] for old_id in old_edge_index[0]])
    edge_index[1] = torch.tensor([id_mapping[old_id.item()] for old_id in old_edge_index[1]])
    print('node swapped')

    def swap_tensor(t):
        # st = torch.empty_like(t)
        st = t[sorted_node_ids]
        return st

    x = swap_tensor(x)
    y = swap_tensor(y)
    split = swap_tensor(split)

    return edge_index, x, y, split


def part_pyg_graph(pyg_graph):
    edge_index, x, y, split = map(pyg_graph.data_dict.get, ['edge_index', 'x', 'y', 'split'])
    return metis_reorder(edge_index, x, y, split)


def main():
    #r = Reddit()
    sr = SmallerReddit()
    edge_index, x, y, split = part_pyg_graph(sr)
    parted_data_path = os.path.join('..', 'torch_ready_data', 'PartedSmallerReddit', 'processed', 'data.pt')
    os.makedirs(os.path.dirname(parted_data_path), exist_ok=True)
    torch.save({"x": x, "y": y, "edge_index": edge_index, "split": split}, parted_data_path)

    # Cora()
    # Amazon()
    pass


if __name__ == '__main__':
    main()