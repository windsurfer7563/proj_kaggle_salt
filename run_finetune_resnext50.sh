#!/bin/bash

mkdir -p data/models/SE_ResNext50/pretrained2/

for i in 0 1 2 3 4
do
   python train.py --config SE_ResNext50_finetune.json  --fold $i --workers 6  --n-epochs 80
done