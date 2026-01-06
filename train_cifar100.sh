export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29508
export DATA_PATH='/home/ruihan/data/cifar-100-images/train'
export MAR_SIZE='mar_base'
# export OUTPUT_DIR="output_dir/${MAR_SIZE}"
export OUTPUT_DIR="output_dir/cifar100/${MAR_SIZE}"
export CACHED_PATH='/home/ruihan/data/cifar-100-images/mar-vae-cached-latents/train'
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_mar.py \
--img_size 32 --bottleneck_dim 768 --patch_size 16 --class_num 100 \
--model ${MAR_SIZE} --diffloss_d 6 --diffloss_w 1024 \
--epochs 2000 --warmup_epochs 100 --batch_size 128 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} \
--data_path ${DATA_PATH} \
--online_eval --eval_bsz 256 \
--cfg 2.9 --cfg_schedule linear  --temperature 1.0 \
--num_iter 4 --num_sampling_steps 100 \
--eval_freq 20 --save_last_freq 5 \
# --wandb \
# --resume ${OUTPUT_DIR} \
# --use_cached --cached_path ${CACHED_PATH} \
