#!/bin/bash
cd /root/workspace/multiphysics_simulation/src/train
python nuclear_thermal_coupling.py --train_which neutron --dim 8 --batchsize 32 --paradigm diffusion --model_type Unet --dataset iter1 --n_dataset 5000 --gradient_accumulate_every 2 --epoches 200000
python nuclear_thermal_coupling.py --train_which solid --dim 8 --batchsize 32 --paradigm diffusion --model_type Unet --dataset iter1 --n_dataset 5000 --gradient_accumulate_every 2 --epoches 200000
python nuclear_thermal_coupling.py --train_which fluid --dim 16 --batchsize 32 --paradigm diffusion --model_type Unet --dataset iter1 --n_dataset 5000 --gradient_accumulate_every 2 --epoches 200000
python nuclear_thermal_coupling.py --train_which neutron --dim 8 --batchsize 32 --paradigm surrogate --model_type Unet --dataset iter1 --n_dataset 5000 --epoches 100000
python nuclear_thermal_coupling.py --train_which solid --dim 8 --batchsize 32 --paradigm surrogate --model_type Unet --dataset iter1 --n_dataset 5000 --epoches 100000
python nuclear_thermal_coupling.py --train_which fluid --dim 16 --batchsize 32 --paradigm surrogate --model_type Unet --dataset iter1 --n_dataset 5000 --epoches 100000
