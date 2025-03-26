CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
--main_process_port 8878 \
--config_file acc_configs/gpu8.yaml main.py objaverse \
--workspace workspace \
--batch_size 8 \
--trainlist pbr_train_list.txt \
--testlist pbr_test_list.txt \
--gradient_accumulation_steps 1 \
--text_description Cap3D_automated_Objaverse_full.csv \
--image_dir path_to_your_image_folder \
--pointcloud_dir path_to_your_pointcloud_folder \
# --resume model.safetensors \