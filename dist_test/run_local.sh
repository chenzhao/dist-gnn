#!/bin/sh

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7; python -m torch.distributed.launch --nproc_
per_node=6 gcn_distr_15d.py --graphname=Reddit
