#!/bin/bash

MODEL_NAME="LoSTer_KL"

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

HIDDEN_SIZE=256
N_ENCODER_LAYERS=3
N_DECODER_LAYERS=3
DROPOUT=0.1
ALPHA=1.0
ALPHA_KL=0.1
TOL=0.001

BATCH_SIZE=128
EPOCHS_PRETRAIN=50
EPOCHS=100
LR=0.001
STEP_SIZE=5

NUM_WORKERS=4

SEEDS="0 1 2 3 4"


for DATASET_NAME in $DATASET_NAMES;
do
	for SEED in $SEEDS;
	do
		CUDA_VISIBLE_DEVICES=0 python main.py --model_name $MODEL_NAME --dataset_name $DATASET_NAME --experiment_name $EXPERIMENT_NAME --hidden_size $HIDDEN_SIZE --n_encoder_layers $N_ENCODER_LAYERS --n_decoder_layers $N_DECODER_LAYERS --dropout $DROPOUT --alpha $ALPHA --alpha_kl $ALPHA_KL --tol $TOL --batch_size $BATCH_SIZE --epochs_pretrain $EPOCHS_PRETRAIN --epochs $EPOCHS --lr $LR --step_size $STEP_SIZE --num_workers $NUM_WORKERS --seed $SEED
	done
done