cd ..

# ----------------------------------------------------------------------------------------------
## use 2x sample duration
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --root_path ./data --video_path ucf/jpg --annotation_path ucf/ucf101_01.json \
--result_path ucf/results_triplet --dataset ucf101 --n_classes 101 --model resnext \
--model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 16 \
--n_threads 32 --checkpoint 5 --n_epochs 400 \

# ----------------------------------------------------------------------------------------------
