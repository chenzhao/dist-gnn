import os
import os.path
import numpy
import scipy.sparse
import torch
from download_torch_ready_data import Reddit, SmallerReddit, OneQuarterReddit, TinyReddit
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

    #print('edge index', edge_index)
    nxg = to_networkx(Data(x=x, edge_index=edge_index))
    #print('partition begin')
    # print('nxg', nxg.nodes)
    # print('nxg', nxg.edges)
    (edgecuts, parts) = metis.part_graph(nxg, P)
    #print('parted', edgecuts, parts)
    node_ids = list(nxg) # only works when no gap in ids
    parted_ids = [(p,old_node_id) for p, old_node_id in zip(parts, node_ids)]
    sorted_node_ids = [node_id for p, node_id in sorted(parted_ids)]
    #print('sorted node ids', sorted_node_ids)

    id_mapping = dict((old_id, new_id) for old_id, new_id in zip(sorted_node_ids, node_ids))
    #print('id mapping', id_mapping)
    old_edge_index = torch.clone(edge_index)

    edge_index[0] = torch.tensor([id_mapping[old_id.item()] for old_id in old_edge_index[0]])
    edge_index[1] = torch.tensor([id_mapping[old_id.item()] for old_id in old_edge_index[1]])
    #print('node swapped', edge_index)

    def swap_tensor(t):
        # st = torch.empty_like(t)
        st = t[sorted_node_ids].clone()
        return st

    x = swap_tensor(x)
    #print(y.size(), y)
    y = swap_tensor(y)
    #print(y.size(), y)
    #print(split.size(), split)
    split = swap_tensor(split)
    #print(split.size(), split)

    return edge_index, x, y, split


def create_parted_dataset(original_dataset):
    original_name = original_dataset.__class__.__name__
    edge_index, x, y, split = metis_reorder(*list(map(original_dataset.data_dict.get, ['edge_index', 'x', 'y', 'split'])))
    parted_data_path = os.path.join('..', 'torch_ready_data', 'Parted'+original_name, 'processed', 'data.pt')
    os.makedirs(os.path.dirname(parted_data_path), exist_ok=True)
    torch.save({"x": x, "y": y, "edge_index": edge_index, "split": split}, parted_data_path)


def main():
    #torch.set_printoptions(threshold=5000)
    # list(map(create_parted_dataset, [SmallerReddit(), OneQuarterReddit(), Reddit()]))
    create_parted_dataset(Reddit())
    # create_parted_dataset(TinyReddit())

    # Cora()
    # Amazon()
    pass


if __name__ == '__main__':
    main()
