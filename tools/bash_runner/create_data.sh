CREATE_DATA='tools/create_data.py'
DATASET_NAME='nuscenes'
ROOT_PATH_PROJ=$(pwd)
ROOT_PATH="--root-path ${ROOT_PATH_PROJ}/data/nuscenes"
OUT_DIR="--out-dir ${ROOT_PATH_PROJ}/data/nuscenes"
EXTRA_TAG='--extra-tag nuscenes'
VERSION='--version v1.0'

DEBUG_PY='-m debugpy --listen 8531 --wait-for-client'

python ${CREATE_DATA} ${DATASET_NAME} ${ROOT_PATH} ${OUT_DIR} ${EXTRA_TAG} ${VERSION}
# python ${DEBUG_PY} ${CREATE_DATA} ${DATASET_NAME} ${ROOT_PATH} ${OUT_DIR} ${EXTRA_TAG} ${VERSION}





