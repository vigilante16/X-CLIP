#!/bin/bash

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M")

# 配置路径和参数
DATA_PATH=/sshfs/datasets/finevideo/
OUTPUT_PATH=ckpts/xclip_finevideo_vit16_${TIMESTAMP}
# MODEL_PATH=/sshfs/pretrains/openai/clip-vit-base-patch16
MODEL_PATH=/sshfs/pretrains/openai/clip-vit-B-16/ViT-B-16.pt
job_name=xclip_finevideo_vit16_${TIMESTAMP}  # 在job_name中加入时间戳

# 记录开始时间
echo "=== 训练开始于 $(date) ===" | tee -a logs/${job_name}

# --init_model ${MODEL_PATH} \
# --cache_dir ${MODEL_PATH} \
# 执行训练脚本，并在关键步骤添加时间戳
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4    --master_port=29502 main_xclip.py \
  --do_train \
  --do_eval \
  --datatype finevideo \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_PATH} \
  --batch_size 16 \
  --batch_size_val 16 \
  --epochs 10 \
  --max_words 64 \
  --max_frames 64 \
  --feature_framerate 1 \
  --coef_lr 1e-3 \
  --slice_framepos 2 \
  --lr 1e-4 \
  --loose_type \
  --sim_header seqTransf \
  --pretrained_clip_name "ViT-B/16" 2>&1 | tee -a logs/${job_name}

# 记录结束时间和总耗时
echo "=== 训练结束于 $(date) ===" | tee -a logs/${job_name}
echo "=== 总耗时: $SECONDS 秒 ===" | tee -a logs/${job_name}
