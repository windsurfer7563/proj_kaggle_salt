#!/bin/bash

mkdir -p data/models/SE_ResNext50/pretrained/

for i in 7
do
   python train.py --config SE_ResNext50_pretrain.json  --fold $i --workers 10  --n-epochs 5 --warmup 1
   python train.py --config SE_ResNext50_pretrain.json  --fold $i --workers 10  --n-epochs 10 --resume 1
   python train.py --config SE_ResNext50_finetune.json  --fold $i --workers 10  --n-epochs 120
   python train.py --config SE_ResNext50_finetune2.json  --fold $i --workers 10 --n-epochs 275

done

#source ./run_finetune_resnext50.sh
#source ./run_finetune2_resnext50.sh