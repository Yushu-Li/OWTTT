#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

STRONG_OOD=$1


python ./OURS.py \
	--dataset ImageNet-R \
	--dataroot ./data \
	--strong_OOD ${STRONG_OOD} \
	--lr 0.001 \
	--delta 0.1 \
	--ce_scale 0.05 \
	--da_scale 0.1

