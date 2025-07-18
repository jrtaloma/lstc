#!/bin/bash

MODEL_NAME="iTransformer"

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
"

D_MODEL=256
DROPOUT=0.1
N_HEADS=8
D_FF=256
E_LAYERS=2
ALPHA=1.0
BETA=0.65
TEMPERATURE=10
TOL=0.001

BATCH_SIZE=128
EPOCHS_PRETRAIN=50
EPOCHS=100
LR=0.0001

NUM_WORKERS=4

SEEDS="0 1 2 3 4"


for DATASET_NAME in $DATASET_NAMES;
do
	for SEED in $SEEDS;
	do
		CUDA_VISIBLE_DEVICES=0 python main.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --d_model $D_MODEL --dropout $DROPOUT --n_heads $N_HEADS --d_ff $D_FF --e_layers $E_LAYERS --alpha $ALPHA --beta $BETA --temperature $TEMPERATURE --tol $TOL --batch_size $BATCH_SIZE --epochs_pretrain $EPOCHS_PRETRAIN --epochs $EPOCHS --lr $LR --num_workers $NUM_WORKERS --seed $SEED
	done
done