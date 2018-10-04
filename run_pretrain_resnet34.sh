#!/bin/bash

mkdir -p data/models/ResNet34/pretrained/

for i in 0 1 2 3 4
do
   python train.py --config ResNet34_pretrain.json  --fold $i --workers 4  --n-epochs 5  --warmup 1
   python train.py --config ResNet34_pretrain.json  --fold $i --workers 4  --n-epochs 35 --resume 1

done