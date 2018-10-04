#!/bin/bash

mkdir -p data/models/SE_ResNext101/pretrained/

for i in 0
do
   python train.py --config SE_ResNext101_pretrain.json  --fold $i --workers 4  --n-epochs 5 --warmup 1
   python train.py --config SE_ResNext101_pretrain.json  --fold $i --workers 4  --n-epochs 30 --resume 1

done

#source ./run_finetune_resnext101.sh
#source ./run_finetune2_resnext101.sh