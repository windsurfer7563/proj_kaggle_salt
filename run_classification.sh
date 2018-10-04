#!/usr/bin/env bash

set -e

LOGDIR=$(pwd)/logs/fold5/

#echo "Training..."
#PYTHONPATH=. python prometheus/dl/scripts/train.py \
#   --model-dir=classification \
#    --config=classification/train_5.yml \
#    --logdir=$LOGDIR \
#    --verbose


LOGDIR=$(pwd)/logs/fold4/

echo "Training 4..."
PYTHONPATH=. python prometheus/dl/scripts/train.py \
    --model-dir=classification \
    --config=classification/train_4.yml \
    --logdir=$LOGDIR \
    --verbose


LOGDIR=$(pwd)/logs/fold3/

echo "Training 3..."
PYTHONPATH=. python prometheus/dl/scripts/train.py \
    --model-dir=classification \
    --config=classification/train_3.yml \
    --logdir=$LOGDIR \
    --verbose

LOGDIR=$(pwd)/logs/fold2/

echo "Training 2..."
PYTHONPATH=. python prometheus/dl/scripts/train.py \
    --model-dir=classification \
    --config=classification/train_2.yml \
    --logdir=$LOGDIR \
    --verbose


LOGDIR=$(pwd)/logs/fold1/

echo "Training 1..."
PYTHONPATH=. python prometheus/dl/scripts/train.py \
    --model-dir=classification \
    --config=classification/train_1.yml \
    --logdir=$LOGDIR \
    --verbose