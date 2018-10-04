#!/bin/bash

mkdir -p data/models/SE_ResNext50/pretrained/

for i in 0 1 2 3 4
do
   python train.py --config SE_ResNext50_pretrain.json  --fold $i --workers 4  --n-epochs 5 --warmup 1
   python train.py --config SE_ResNext50_pretrain.json  --fold $i --workers 4  --n-epochs 30 --resume 1

done

