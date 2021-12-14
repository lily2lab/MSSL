#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
cd ../..
savepath='results/cmu/MSSL_cmu_long_term/v3'
modelpath='checkpoints/cmu/MSSL_cmu_long_term/v3'
pretrain_modelpath='checkpoints/cmu/MSSL_cmu_long_term/v2/model.ckpt-240'
logname='logs/cmu/train_MSSL_cmu_long_term.log'
nohup python -u train_MSSL_cmu.py \
    --is_training True \
    --train_data_paths data/cmu_ske/train_cmu_35.npy \
    --valid_data_paths data/cmu_ske/train_cmu_35.npy \
    --test_data_paths data/cmu_ske/ \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 35 \
    --filter_size 3 \
    --min_err  100.0 \
    --num_hidden 64 \
    --encoder_length 1 \
    --decoder_length 1 \
    --Tu_length 2 \
    --lr 0.0001  \
    --batch_size 16 \
    --sampling_stop_iter 0 \
    --max_iterations 300000 \
    --display_interval 10 \
    --test_interval 20 \
    --n_gpu 1 \
    --snapshot_interval 20 >>${logname}  2>&1 & 
tail -f ${logname}
# --pretrained_model ${pretrain_modelpath}  \



