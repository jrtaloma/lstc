#!/bin/bash

MODEL_NAME="DTCR"

EXPERIMENT_NAME="default"

DATASET_NAMES="
CinCECGTorso
NonInvasiveFetalECGThorax1
NonInvasiveFetalECGThorax2
StarLightCurves
UWaveGestureLibraryX
UWaveGestureLibraryY
MixedShapesRegularTrain
MixedShapesSmallTrain
EOGVerticalSignal
SemgHandMovementCh2
ECG5000
OSULeaf
Symbols
MiddlePhalanxOutlineAgeGroup
ProximalPhalanxOutlineAgeGroup
ProximalPhalanxTW
SyntheticControl
m5-forecasting-accuracy
"

WEIGHT=1.0
TOL=0.001

BATCH_SIZE=128
EPOCHS=100
LR=0.005
ALTER_ITER=10

NUM_WORKERS=4

SEEDS="0 1 2 3 4"


for DATASET_NAME in $DATASET_NAMES;
do
	for SEED in $SEEDS;
	do
		CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $DATASET_NAME --weight $WEIGHT --tol $TOL --batch_size $BATCH_SIZE --epochs $EPOCHS --lr $LR --alter_iter $ALTER_ITER --num_workers $NUM_WORKERS --seed $SEED
	done
done