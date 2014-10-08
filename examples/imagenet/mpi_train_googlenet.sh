#!/usr/bin/env sh
GOOGLE_LOG_DIR=models/googlenet/log \
mpirun -np 2 ./build/tools/caffe train \
    --solver=models/googlenet/solver.prototxt 2>&1 | tee ./models/googlenet/log_googlenet.txt \

