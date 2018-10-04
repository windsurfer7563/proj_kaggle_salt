#!/bin/bash

for i in 0
do
   python train.py --config IncV3_finetune.json --fold $i --workers 4  --n-epochs 190

done