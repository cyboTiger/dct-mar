export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export CACHED_PATH='/home/ruihan/data/mar-vae-cached'
export IMAGENET_PATH='/home/ruihan/data/imagenet'

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_cache.py \
--num_workers 24 --pin_mem --cached_path ${CACHED_PATH} --data_path ${IMAGENET_PATH}