#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
cd ../..
savepath='results/3dpw/MSSL_3dpw_long_term/v3'
modelpath='checkpoints/3dpw/MSSL_3dpw_long_term/v3'
pretrain_modelpath='checkpoints/3dpw/MSSL_3dpw_long_term/v1/model.ckpt-2300'
logname='logs/3dpw/train_MSSL_3dpw_long_term.log'
nohup python -u train_MSSL_3dpw.py \
    --is_training True \
    --dataset_name skeleton \
    --train_data_paths data/3dpw_ske/train_3dpw0_40.npy \
    --valid_data_paths data/3dpw_ske/test_3dpw0_40.npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 40 \
    --filter_size 3 \
    --min_err  1000.0 \
    --num_hidden 64 \
    --encoder_length 1 \
    --decoder_length 1 \
    --Tu_length 2 \
    --lr 0.0001  \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 300000 \
    --display_interval 10 \
    --test_interval 100 \
    --n_gpu 1 \
    --snapshot_interval 100 >>${logname}  2>&1 & 
tail -f ${logname}
# --pretrained_model ${pretrain_modelpath}  \



