
CONFIG_FILE='myprojects/megvii_dataset/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs_megvii.py'
TRAIN_PY='tools/train.py'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'
GPU_NUM=8

# 多卡训练
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

# 用法：
# sh myprojects/megvii_dataset/bash_runner/train.sh