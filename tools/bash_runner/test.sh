
TEST_PY='tools/test.py'
CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py'
PTH='ckpt/voxel0075_vov_1600x640_epoch20.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval bbox
# python ${DEBUG_PY} ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval bbox
