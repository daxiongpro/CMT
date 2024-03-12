
TEST_PY='tools/test.py'
CONFIG_FILE='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py'
PTH='ckpt/voxel0075_vov_1600x640_epoch20.pth'
DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

# 测试 mAP 等指标
# python ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval bbox
# python ${DEBUG_PY} ${TEST_PY} ${CONFIG_FILE} ${PTH} --eval bbox

# 测试 nuscenes 测试集，并保存 nuscenes 格式结果
python ${TEST_PY} ${CONFIG_FILE} ${PTH} --format-only --eval-options jsonfile_prefix=test/
