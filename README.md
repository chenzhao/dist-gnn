This repo contains various experimental code for our distributed GNN works.



For easy-to-use code: https://github.com/chenzhao/dist-gnn/tree/main/light_dist_gnn








##### For AutoRepeat: Adaptive CommUnicaTiOn REduction for Distributed GraPh NEurAl NeTworks
check 1D_CAGNET

- download datasets
```
./download_torch_ready_data.py
``` 
- preprocess graph data
```
./coo_graph.py
``` 
- preprocess graph data
```
./coo_graph.py
```
- run experiments
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port 23456 dist_1d.py --graphname=Reddit --world_size=4 --epochs=300
```


##### Adaptive bcast result:

https://docs.google.com/presentation/d/1oT5A3Omih0HIg2zS5E8fsN9W2-zKtL8j5f-eb2XJzWc/edit?usp=sharing
