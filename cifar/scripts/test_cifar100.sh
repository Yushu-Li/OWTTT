#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)


CORRUPT=$1
STRONG_OOD=$2



python TEST.py \
	--dataset cifar100OOD \
	--dataroot ./data \
	--strong_OOD ${STRONG_OOD} \
	--resume ./results/cifar100_joint_resnet50 \
	--corruption ${CORRUPT} 

