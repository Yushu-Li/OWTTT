#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

CORRUPT=$1
STRONG_OOD=$2


python ./OURS.py \
	--dataset ImageNet-C \
	--dataroot ./data \
	--strong_OOD ${STRONG_OOD} \
	--corruption ${CORRUPT} \
	--lr 0.001 \
	--delta 0.1 \
	--ce_scale 0.05 \
	--da_scale 0.1