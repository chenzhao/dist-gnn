# Partition Test

These scripts aim to profile the training process of GNN models on multiple cards,
including the computation time, communication time, and communication data size.


## Environment Setup Using Anaconda on LCCPU servers

This setting is based on cuda-10.2 due to some API compatibilities.

1. set some environment variables

`export CUDA_HOME=/usr/local/cuda-10.2/
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"`

2. install new gcc

`conda install libgcc`

3. install tmux (optional)

`conda install -c conda-forge tmux`

4. install pytorch and cudatoolkit

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

5. install the sparse extension

Go to 1D_CAGNET/sparse-extension and execute `python setup.py install`.

## Run

Run this command to train a GCN model on 4 cards, 300 epochs. Each card will produce two pkl files in 'logs' directory.

`python -m torch.distributed.launch --nproc_per_node=4 --master_port 23456 dist_1d.py --graphname=Reddit --world_size=4 --epochs=300`


## Plot

Plot and **show** the profile results: `python read_log.py`.

Change the code if you want to get figure files directly.
