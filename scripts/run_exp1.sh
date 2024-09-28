#!/bin/bash
cd /root/workspace/multiphysics_simulation/src/train
python reaction_diffusion.py --train_which u --dim 24 --batchsize 256 --paradigm diffusion --model_type Unet --epoches 200000
python reaction_diffusion.py --train_which v --dim 24 --batchsize 256 --paradigm diffusion --model_type Unet --epoches 200000
python reaction_diffusion.py --train_which u --dim 24 --batchsize 256 --paradigm surrogate --model_type Unet --epoches 100000
python reaction_diffusion.py --train_which v --dim 24 --batchsize 256 --paradigm surrogate --model_type Unet --epoches 100000
