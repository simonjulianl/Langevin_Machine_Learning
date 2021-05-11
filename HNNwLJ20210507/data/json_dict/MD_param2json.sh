#!/bin/bash

# run like this ../data/gen_by_MC/train/xxx, xxx
# $1,$2 : MC_output basename, basename

echo "run MD sampler use MD parameters w ML parameters"
echo "show MC output basename $1 to load data for MD, create folder name $2 in gen_by_MD"
# usage <programe> <MC_init_dir+basename> <basename> <hamiltonian_type : noML or pair_wise_HNN> <json_dir>
CUDA_VISIBLE_DEVICES='' python MD_param2json.py $1 $2 noML ../gen_by_MD/

#usage <programe> <train_dir+filename> <valid_dir+filename> <basename> <json_dir>
CUDA_VISIBLE_DEVICES='' python  ML_param2json.py 'None' 'None' $2 ../gen_by_MD/

#
#
