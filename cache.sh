export RANK=0
export WORLD_SIZE=0
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export CACHED_PATH='/home/ruihan/data/mar-vae-cached-latents/cifar-100/test'
export DATA_PATH='/home/ruihan/data/cifar-100-images'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_cache.py \
--img_size 32 \
--num_workers 48 --pin_mem --cached_path ${CACHED_PATH} --data_path ${DATA_PATH} \
--pin_mem --batch_size 256