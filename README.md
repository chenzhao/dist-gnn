


# AutoRepeat: Adaptive CommUnicaTiOn REduction for Distributed GraPh NEurAl NeTworks


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


## dist-gnn


The sparse matrix operation in 1D_CAGNET is [SC20] code simplified.

Added some graph data cache and removed PyG dependency. 

Graph data cache will be created at the first time run. 

Adaptive bcast result:

https://docs.google.com/presentation/d/1oT5A3Omih0HIg2zS5E8fsN9W2-zKtL8j5f-eb2XJzWc/edit?usp=sharing
