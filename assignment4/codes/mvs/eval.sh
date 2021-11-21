#!/usr/bin/env bash
DTU_TESTING="./dtu_dataset/"
CKPT_FILE="./checkpoints/model_000003.ckpt"
python eval.py --dataset=dtu_eval --batch_size=1 --testpath=$DTU_TESTING \
--testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
