#!/bin/bash

#mkdir -p data/models/SE_ResNext50/pretrained/


for i in  0
do
   #python train.py --config SE_ResNext50_2_finetune.json  --fold $i --workers 10  --n-epochs 10 --warmup 1
   python train.py --config SE_ResNext50_2_finetune.json  --fold $i --workers 10  --n-epochs 60
   python train.py --config SE_ResNext50_2_finetune2.json  --fold $i --workers 10 --n-epochs 335
done

#source ./run_finetune_resnext50.sh
#source ./run_finetune2_resnext50.sh2