#!/bin/bash

mkdir -p data/models/SE_ResNext101/pretrained2/

for i in 0
do
   python train.py --config SE_ResNext101_finetune.json  --fold $i --workers 12  --n-epochs 120
done