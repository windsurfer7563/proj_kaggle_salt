#!/bin/bash


for i in 0 3
do
   python train.py --config SE_ResNext50_finetune2.json  --fold $i --workers 6 --n-epochs 250

done