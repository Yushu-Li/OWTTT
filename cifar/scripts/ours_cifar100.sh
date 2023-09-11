#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

CORRUPT=$1
STRONG_OOD=$2

python OURS.py \
	--dataset cifar100OOD \
	--dataroot ./data \
	--strong_OOD ${STRONG_OOD} \
	--resume ./results/cifar100_joint_resnet50 \
	--corruption ${CORRUPT} \
	--lr 0.001 \
	--delta 0.1 \
	--da_scale 1 \
	--ce_scale 0.2

