export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export IMAGENET_PATH='/home/ruihan/data/imagenet'
export MAR_SIZE='mar_base'
export OUTPUT_DIR="output_dir/${MAR_SIZE}"
export CACHED_PATH='/home/ruihan/data/mar-vae-cached'

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_mar.py \
--img_size 256 --vae_path pretrained_models/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model ${MAR_SIZE} --diffloss_d 6 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 128 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} \
--online_eval --eval_bsz 256 \
--cfg 2.9 --cfg_schedule linear  --temperature 1.0 \
--num_iter 64 --num_sampling_steps 100 \
--eval_freq 50 --save_last_freq 5
# --use_cached --cached_path ${CACHED_PATH} \
# --resume ${OUTPUT_DIR}