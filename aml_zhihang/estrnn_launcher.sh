#!/bin/bash

# update pip
/opt/conda/bin/python -m pip install --upgrade pip

echo $HOME
#export PYTHONPATH=$PYTHONPATH:$HOME
export PYTHONIOENCODING=utf-8
locale -a

# If install some libs, write after here
pip install lmdb
pip install lpips
pip install tqdm
pip install thop
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# for logging and checkpointing
IFS='-'
read -a proj_name_arr <<< "$AZUREML_ARM_PROJECT_NAME"
read -a run_id_arr <<< "$AZUREML_RUN_ID"
project_name=${proj_name_arr[0]}
nid=${#strarr[*]}
job_name=${run_id_arr[nid-2]}
exp_name=${run_id_arr[nid-3]}
prefix=/mnt/output/projects/${project_name}/${exp_name}

IFS=' '
SAVE_DIR=${prefix}/${job_name}/amlt_results
DATA_DIR=/mnt/input/low_level/
mkdir -p $SAVE_DIR
echo $HOME
echo $DATA_DIR
echo $SAVE_DIR
echo "node rank"
echo $NODE_RANK

python main.py --data_root $DATA_DIR --dataset BSD --save_dir $SAVE_DIR --ds_config 2ms16ms

