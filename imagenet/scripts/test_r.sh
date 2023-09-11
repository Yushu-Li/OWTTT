#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
STRONG_OOD=$1


python ./TEST.py \
	--dataset ImageNet-R \
	--dataroot ./data \
	--strong_OOD ${STRONG_OOD}

