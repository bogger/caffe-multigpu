#!/usr/bin/env sh
# Hongyang LI, v4
# change data to compact_data; 
# perform multi-scale training (as Xiong's version); 
# change lr_policy;

MPI_PATH=/usr/bin/

#--weights=googlenet_train_v3.1_iter_595000.caffemodel \
# MPI implementation
GLOG_logtostderr=1 
../../build/tools/caffe train \
	--solver=solver_v4_ext_label.prototxt \
2>&1 | tee googlenet_v4_ext_label.log 