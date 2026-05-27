#!/bin/bash

set -e

echo "============================================================"
echo "CIFAR-10 Big Table Experiments"
echo "Methods: ERM / CutMix / CutMix+SUMix"
echo "Backbone: ResNet18"
echo "Epochs: 200"
echo "============================================================"

DATASET="cifar10"
MODEL="resnet18"
BATCH_SIZE=128
EPOCHS=200
LR=0.1
OPTIMIZER="sgd"
MOMENTUM=0.9
WEIGHT_DECAY=5e-4
LR_SCHEDULE="multistep"
LR_MILESTONES="100,150"
LR_GAMMA=0.1
SEED=0
DATA_DIR="./data"
OUTPUT_DIR="./outputs"

CUTMIX_ALPHA=1.0
CUTMIX_PROB=1.0
SUMIX_GAMMA=0.1

mkdir -p logs

echo ""
echo "============================================================"
echo "Run 1/3: ERM / no augmentation"
echo "============================================================"

python train.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --aug none \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --optimizer ${OPTIMIZER} \
  --momentum ${MOMENTUM} \
  --weight-decay ${WEIGHT_DECAY} \
  --lr-schedule ${LR_SCHEDULE} \
  --lr-milestones ${LR_MILESTONES} \
  --lr-gamma ${LR_GAMMA} \
  --seed ${SEED} \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUTPUT_DIR} \
  2>&1 | tee logs/cifar10_resnet18_none_seed${SEED}.log


echo ""
echo "============================================================"
echo "Run 2/3: CutMix"
echo "============================================================"

python train.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --aug cutmix \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --optimizer ${OPTIMIZER} \
  --momentum ${MOMENTUM} \
  --weight-decay ${WEIGHT_DECAY} \
  --lr-schedule ${LR_SCHEDULE} \
  --lr-milestones ${LR_MILESTONES} \
  --lr-gamma ${LR_GAMMA} \
  --cutmix-alpha ${CUTMIX_ALPHA} \
  --cutmix-prob ${CUTMIX_PROB} \
  --seed ${SEED} \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUTPUT_DIR} \
  2>&1 | tee logs/cifar10_resnet18_cutmix_seed${SEED}.log


echo ""
echo "============================================================"
echo "Run 3/3: CutMix + SUMix"
echo "============================================================"

python train.py \
  --dataset ${DATASET} \
  --model ${MODEL} \
  --aug cutmix_sumix \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --optimizer ${OPTIMIZER} \
  --momentum ${MOMENTUM} \
  --weight-decay ${WEIGHT_DECAY} \
  --lr-schedule ${LR_SCHEDULE} \
  --lr-milestones ${LR_MILESTONES} \
  --lr-gamma ${LR_GAMMA} \
  --cutmix-alpha ${CUTMIX_ALPHA} \
  --cutmix-prob ${CUTMIX_PROB} \
  --sumix-gamma ${SUMIX_GAMMA} \
  --seed ${SEED} \
  --data-dir ${DATA_DIR} \
  --output-dir ${OUTPUT_DIR} \
  2>&1 | tee logs/cifar10_resnet18_cutmix_sumix_seed${SEED}.log


echo ""
echo "============================================================"
echo "All experiments finished."
echo "CSV files should be under:"
echo "${OUTPUT_DIR}/${DATASET}/${MODEL}/"
echo ""
echo "Logs are under:"
echo "./logs/"
echo "============================================================"