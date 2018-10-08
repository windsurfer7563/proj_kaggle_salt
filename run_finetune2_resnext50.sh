#!/bin/bash


for i in 1
do
   python train.py --config SE_ResNext50_finetune2.json  --fold $i --workers 6 --n-epochs 275

done