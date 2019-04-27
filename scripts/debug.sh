cd ..


# ----------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python main.py --root_path ./data --video_path hmdb/jpg --annotation_path hmdb/hmdb51_1.json \
--result_path hmdb/results --dataset hmdb51 --model resnet \
--model_depth 34 --n_classes 51 --batch_size 32 --n_threads 0 --checkpoint 5

# ----------------------------------------------------------------------------------------------
