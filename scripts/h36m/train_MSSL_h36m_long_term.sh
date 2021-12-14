#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
cd ../..
savepath='results/h36m/MSSL_h36m_long_term/v0'
modelpath='checkpoints/h36m/MSSL_h36m_long_term/v0'
pretrain_modelpath='checkpoints/h36m/MSSL_h36m_long_term/v0/model.ckpt-20'
logname='logs/h36m/train_MSSL_h36m_long_term_v0.log'
nohup python -u train_MSSL_h36m.py \
    --is_training True \
    --train_data_paths data/h36m/h36m_train_3d.npy \
    --valid_data_paths data/h36m/h36m_val_3d.npy \
    --test_data_paths data/h36m/test_npy \
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



