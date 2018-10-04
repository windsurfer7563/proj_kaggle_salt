#!/usr/bin/env bash

set -e

LOGDIR=$(pwd)/logs/fold1/

echo "Inference..."
PYTHONPATH=. python prometheus/dl/scripts/inference.py \
   --model-dir=classification \
   --resume=$LOGDIR/checkpoint.best.pth.tar \
   --out-prefix=$LOGDIR/dataset.predictions.{suffix}.npy \
   --config=$LOGDIR/config.json,./classification/inference.yml \
   --verbose


