#!/bin/bash

MODEL_NAME="PathFormer"

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

NUM_LAYERS=3
K=2
D_MODEL=4
D_FF=64
RESIDUAL_CONNECTION=1
REVIN=0
ALPHA=1.0
BETA=0.65
TEMPERATURE=10
TOL=0.001

BATCH_SIZE=128
EPOCHS_PRETRAIN=10
EPOCHS=30
LR=0.0001

NUM_WORKERS=4

SEEDS="0 1 2 3 4"


for DATASET_NAME in $DATASET_NAMES;
do
	for SEED in $SEEDS;
	do
		CUDA_VISIBLE_DEVICES=0 python main.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --num_layers $NUM_LAYERS --k $K --d_model $D_MODEL --d_ff $D_FF --residual_connection $RESIDUAL_CONNECTION --revin $REVIN --alpha $ALPHA --beta $BETA --temperature $TEMPERATURE --tol $TOL --batch_size $BATCH_SIZE --epochs_pretrain $EPOCHS_PRETRAIN --epochs $EPOCHS --lr $LR --num_workers $NUM_WORKERS --seed $SEED
	done
done