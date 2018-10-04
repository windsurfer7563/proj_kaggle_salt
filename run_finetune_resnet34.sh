#!/bin/bash

for i in 0 1 2 3 4
do
   python train.py --config ResNet34_finetune.json --fold $i --workers 4  --n-epochs 125
   done