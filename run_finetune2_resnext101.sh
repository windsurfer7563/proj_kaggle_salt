#!/bin/bash


for i in 0
do
   python train.py --config SE_ResNext101_finetune2.json  --fold $i --workers 12 --n-epochs 250

done