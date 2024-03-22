if [ "$1" = "megvii" ]; then
    CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs_megvii.py'
else
    CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py'  # OOM on RTX 3090 with batch_size=2
    # CONFIG_FILE='projects/configs/fusion/cmt_voxel0100_r50_800x320_cbgs.py'
fi

TRAIN_PY='tools/train.py'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'
GPU_NUM=1

# 多卡训练
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}

# 用法：
# sh tools/bash_runner/train.sh         # 训练 nuscenes 数据集
# sh tools/bash_runner/train.sh megvii  # 训练 megvii 数据集