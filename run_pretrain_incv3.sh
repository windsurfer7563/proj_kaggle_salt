#!/bin/bash

mkdir -p data/models/IncV3/pretrained/

for i in 0 1 2 3 4
do
   python train.py --config IncV3_pretrain.json   --fold $i --workers 4  --n-epochs 10  --warmup 1
   python train.py --config IncV3_pretrain.json  --fold $i --workers 4  --n-epochs 40 --resume 1

done