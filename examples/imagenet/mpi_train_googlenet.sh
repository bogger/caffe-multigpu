#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/googlenet/log \
mpirun -np 2 ./build/tools/caffe train \
    --solver=models/googlenet/solver.prototxt \
#    --snapshot=models/googlenet/googlenet_train_iter_210000.solverstate
#    --gpu=3
