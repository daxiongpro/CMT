TRAIN_PY='tools/train.py'
CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py'  # OOM on RTX 3090 with batch_size=2
# CONFIG_FILE='projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'
GPU_NUM=8


# 多卡训练
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}