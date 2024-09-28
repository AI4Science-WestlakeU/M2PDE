#!/bin/bash
cd /root/workspace/multiphysics_simulation/src/train
python heatpipe.py --batchsize 256 --model_type transformer --paradigm surrogate --n_layer 5 --hidden_dim 64 --epoches 100000
python heatpipe.py --batchsize 256 --model_type transformer --paradigm diffusion --n_layer 5 --hidden_dim 64 --epoches 200000
