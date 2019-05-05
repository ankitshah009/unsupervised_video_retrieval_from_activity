cd ..

# ----------------------------------------------------------------------------------------------
## use 2x sample duration
# run this for debugging
# CUDA_VISIBLE_DEVICES=1,2,3 python main.py --root_path ./data --video_path hmdb/jpg --annotation_path hmdb/hmdb51_1.json \
# --result_path hmdb/results_triplet_scratch --dataset hmdb51 --n_classes 51 --model resnext \
# --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 8 \
# --n_threads 32 --checkpoint 5 --n_epochs 250 --no_val \

# ----------------------------------------------------------------------------------------------
# run this for actual
CUDA_VISIBLE_DEVICES=1,2,3 python main.py --root_path ./data --video_path hmdb/jpg --annotation_path hmdb/hmdb51_1.json \
--result_path hmdb/results_triplet_scratch --dataset hmdb51 --n_classes 51 --model resnext \
--model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64 --batch_size 16 \
--n_threads 32 --checkpoint 5 --n_epochs 250 \
