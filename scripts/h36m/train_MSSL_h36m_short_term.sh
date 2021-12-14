#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
cd ../..
savepath='results/h36m/MSSL_h36m_short_term/v3'
modelpath='checkpoints/h36m/MSSL_h36m_short_term/v3'
pretrain_modelpath='checkpoints/h36m/MSSL_h36m_short_term/v2/model.ckpt-3100'
logname='logs/h36m/train_MSSL_h36m_short_term.log'
nohup python -u train_MSSL_h36m.py \
    --is_training True \
    --train_data_paths data/h36m20/h36m20_train_3d.npy \
    --valid_data_paths data/h36m20/h36m20_val_3d.npy \
    --test_data_paths data/h36m20/test20_npy \
    --save_dir ${modelpath} \
    --gen_dir ${savepath} \
    --input_length 10 \
    --seq_length 20 \
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



