#!/bin/bash

# generate different nsamples each temperature; high T makes more nsamples

# run like this ./MC_run.sh 4 3325 1
# $1,$2,$3: nparticle, seed nsamples

# 4 particles : T0.04 -> 0.018, T0.16 -> 0.035, T0.32 -> 0.08  
# 8 particles :       -> 0.01         -> 0.023,       -> 0.04 
# 16 particles :      -> 0.006        -> 0.015        -> 0.02

#for i in 3251 5928 1371 4729; do # train
#dq=(0.018 0.02 0.03  0.04 0.05 0.06 0.07 0.08 0.1) # 4 particles
#dq=(0.01 0.014 0.02  0.025 0.05 0.06 0.07 0.08 0.1) # 8 particles
dq=(0.008 0.01 0.013  0.025 0.05 0.06 0.07 0.08 0.1) # 16 particles
temp=(0.03 0.11 0.19 0.27 0.35 0.43 0.51 0.59 0.67)
#temp=$(seq 0.03 0.08 0.69)
for i in $(seq 0 8); do  

        echo "n$1 T${temp[$i]} seed $2 nsamples $3 dq ${dq[$i]}"

	#usage <programe> <nparticle> <temperature> <seed> <nsamples> <dq>
	CUDA_VISIBLE_DEVICES='' python  MC_param2json.py $1 ${temp[$i]} $2 $3 ${dq[$i]}
done
