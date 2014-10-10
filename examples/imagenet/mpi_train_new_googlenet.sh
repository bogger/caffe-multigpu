#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/googlenet \
mpirun -np 4 ./build/tools/caffe train \
    --solver=models/new_googlenet/solver.prototxt \
#    --gpu=3
