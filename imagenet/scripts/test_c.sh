#! /usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
CORRUPT=$1
STRONG_OOD=$2


python ./TEST.py \
	--dataset ImageNet-C \
	--dataroot ./data \
	--strong_OOD ${STRONG_OOD} \
	--corruption ${CORRUPT}


