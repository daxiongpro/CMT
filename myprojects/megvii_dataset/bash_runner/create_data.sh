# conda activate bevfusion

ROOT_PATH_PROJ=$(pwd)
ROOT_PATH_DATASET=${ROOT_PATH_PROJ}'data/megvii_data/new_custom_data'
echo ${ROOT_PATH_DATASET}
CREATE_DATA_PY="myprojects/megvii_dataset/tools/create_data.py"
python ${CREATE_DATA_PY} megvii --root-path ${ROOT_PATH_DATASET} --out-dir ${ROOT_PATH_DATASET} --extra-tag megvii