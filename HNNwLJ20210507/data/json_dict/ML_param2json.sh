#!/bin/bash

# $1,$2,$3 : ../data/gen_by_MD/train/xxxx.pt, ../data/gen_by_MD/valid/xxxx.pt, xxxx
# $4: pairwise_HNN

echo "run ML trainer use ML parameters w MD parameters"
echo "show train valid data filenames, $1 and $2. create folder name $3 in gen_by_ML"

#usage <programe> <train_dir+filename> <valid_dir+filename> <basename> <json_dir>
CUDA_VISIBLE_DEVICES='' python  ML_param2json.py $1 $2 $3 ../gen_by_ML/

# usage <programe> <MC_init_dir+basename> <basename> <hamiltonian_type : noML or pairwise_HNN> <json_dir>
CUDA_VISIBLE_DEVICES='' python MD_param2json.py 'None' $3 $4 ../gen_by_ML/

#
#
