cd ..

# ----------------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=1 python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
# --result_path results --dataset kinetics --model resnet \
# --model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5

# ----------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ./data --video_path hmdb/jpg --annotation_path hmdb/hmdb51_1.json \
--result_path hmdb/results --dataset hmdb51 --model resnet \
--model_depth 34 --n_classes 51 --batch_size 128 --n_threads 4 --checkpoint 5

# ----------------------------------------------------------------------------------------------