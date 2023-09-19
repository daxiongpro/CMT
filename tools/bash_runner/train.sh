DATE=$(date '+%Y-%m-%d_%H-%M-%S')
TRAIN_PY='tools/train.py'
CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py'
WORK_DIR="runs/${DATE}/"
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

python ${TRAIN_PY} ${CONFIG_FILE}
# python ${DEBUG_PY} ${TRAIN_PY} ${CONFIG_FILE}
