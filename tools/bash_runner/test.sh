
TEST_PY='tools/test.py'
CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py'
PTH='ckpts/voxel0075_vov_1600x640_epoch20.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'
GPU_NUM=1

# 单块显卡测试
python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval bbox
# python ${DEBUG_PY} ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval bbox

# 多块显卡测试
# tools/dist_test.sh ${CONFIG_FILE} ${PTH} ${GPU_NUM} --eval bbox
# tools/dist_test.sh ${CONFIG_FILE} ${PTH} ${GPU_NUM} --format-only --eval-options jsonfile_prefix=test/