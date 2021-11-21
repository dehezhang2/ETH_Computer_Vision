#!/usr/bin/env bash
MVS_TRAINING="./dtu_dataset/"
python train.py --dataset=dtu --batch_size=2 --epochs 4 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/val.txt --numdepth=192 --logdir ./checkpoints $@
