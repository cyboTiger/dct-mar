export DATA_PATH='/home/ruihan/data/cifar-100-images/test'
export MAR_SIZE='mar_base'

# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
python \
main_mar.py \
--img_size 32 \
--model ${MAR_SIZE} --diffloss_d 6 --diffloss_w 1024 \
--eval_bsz 256 --num_images 50000 \
--num_iter 4 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0  \
--output_dir output_dir/${MAR_SIZE} \
--data_path ${DATA_PATH} --evaluate \
--resume output_dir/${MAR_SIZE}