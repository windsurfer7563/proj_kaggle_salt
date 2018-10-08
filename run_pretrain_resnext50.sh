#!/bin/bash

mkdir -p data/models/SE_ResNext50/pretrained/

for i in 1
do
   python train.py --config SE_ResNext50_pretrain.json  --fold $i --workers 4  --n-epochs 5 --warmup 1
   python train.py --config SE_ResNext50_pretrain.json  --fold $i --workers 4  --n-epochs 30 --resume 1

done

source ./run_finetune_resnext50.sh
source ./run_finetune2_resnext50.sh