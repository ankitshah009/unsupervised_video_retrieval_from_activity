cd ..

# ----------------------------------------------------------------------------------------------
## use 2x sample duration
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --root_path ./data --video_path hmdb/jpg --annotation_path hmdb/hmdb51_1.json \
--result_path hmdb/results_triplet_scratch --dataset hmdb51 --n_classes 51 --model resnext \
--model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 16 \
--n_threads 32 --checkpoint 5 --n_epochs 250 \

# ----------------------------------------------------------------------------------------------
