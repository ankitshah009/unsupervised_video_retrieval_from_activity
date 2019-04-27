cd ..

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
## HMDB
# python utils/video_jpg_ucf101_hmdb51.py ./data/raw_data/hmdb ./data/raw_data/hmdb
# python utils/n_frames_ucf101_hmdb51.py ./data/hmdb
# python utils/hmdb51_json.py ./data/raw_data/testTrainMulti_7030_splits

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
## UCF
# python utils/video_jpg_ucf101_hmdb51.py ./data/raw_data/UCF-101 ./data/raw_data/UCF-101
# python utils/n_frames_ucf101_hmdb51.py ./data/ucf
python utils/ucf101_json.py ./data/raw_data/ucfTrainTestlist
