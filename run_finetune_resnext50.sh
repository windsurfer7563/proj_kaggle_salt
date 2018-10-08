#!/bin/bash

mkdir -p data/models/SE_ResNext50/pretrained2/

for i in 1
do
   python train.py --config SE_ResNext50_finetune.json  --fold $i --workers 6  --n-epochs 90
done