CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--main_process_port 8878 \
--config_file acc_configs/gpu8.yaml main.py shapenet \
--workspace workspace \
--batch_size 16 \
--trainlist shapenet_filelist/train_bench.txt \
--testlist shapenet_filelist/test_bench.txt \
--gradient_accumulation_steps 1 \
--image_dir path_to_your_image_folder \
--pointcloud_dir path_to_your_pointcloud_folder \
# --resume model.safetensors \
