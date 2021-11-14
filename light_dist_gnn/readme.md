
## A light version of distributed GNN training environment


#### Code improvements compared with SC20 and original dist-gnn

- SpMM for cuda11. cusparseScsrmm2 is deprecated, for cuda11 we need cusparseSpMM. CUSPARSE_SPMM_COO_ALG4 is used by default because, according to tests with 2080ti and CUDA11.1, other SPMM ALG is much slower.
- A unified entry. Run main.py to start all workers then, supposedly, they terminate together if any error happens. 
- Useless code removed.
- Well-structured.


#### Other improvements
- Padding for last block. When num_nodes is not divisible by n_procs, some dummy nodes with no neighbors are added to make equal tensor sizes. 
- Unified dataset management. No pyg and dgl dependency for training.
- More models. A cached GCN. A partially parallel GAT.
- Slightly simpler logger and timer.


### Get started
##### Create an environment.
For cuda10,
install pytorch for training (copied from pytorch page):
```
conda create --name gnn
conda activate gnn
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
```
if needed, install other frameworks to download datasets:
```
conda install -c dglteam dgl-cuda10.2
conda install pyg -c pyg -c conda-forge
pip install ogb
```

Similarly, for cuda11:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install -c dglteam dgl-cuda11.1
conda install pyg -c pyg -c conda-forge
pip install ogb
```
##### Prepare datasets.
Modify prepare_data.py according to the datasets and partition you need. Then
```
python ./prepare_data.py
```
##### Training
main.py handles dist env and shell args.

dist_train.py handles training and evaluating flows.

models define models.
```
python ./main.py
```



##### To use SpMM of cusparse (usually not necessary, but faster).
For old pytorch versions without sparse-tensor-supported addmm (very rare), this is necessary.

Build and install the cpp ext (with compilers and cuda paths in env):
```
cd spmm_cpp
python setup.py install
```


To install compilers in a conda env:
```
conda install gxx_linux-64 
```


To set cuda paths (this is just a googled example, find appropriate settings for your own env):
```
export CUDA_HOME=/usr/local/cuda-11/
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
```

