#!/bin/bash
# generate different nsamples each temperature; high T makes more nsamples

# run like this ./MC_param2json.sh 4
# $1 : nparticle

#dq=(0.018 0.02 0.03  0.04 0.05 0.06 0.07 0.08 0.1) # 4 particles
#dq=(0.01 0.014 0.02  0.025 0.05 0.06 0.07 0.08 0.1) # 8 particles
dq=(0.008 0.01 0.013  0.025 0.05 0.06 0.07 0.08 0.1) # 16 particles
temp=(0.03 0.11 0.19 0.27 0.35 0.43 0.51 0.59 0.67)
nsamples=(4 4 4 8 8 8 12 12 12)
for i in 3251 5928 1371 4729 7483 8976; do

        for j in $(seq 0 8); do

                echo "n$1 T${temp[$j]} seed $i nsamples ${nsamples[$j]} dq ${dq[$j]}"

                #usage <programe> <nparticle> <temperature> <seed> <nsamples> <dq>
                CUDA_VISIBLE_DEVICES='' python  MC_param2json.py  $1  ${temp[$j]} $i ${nsamples[$j]} ${dq[$j]}
        done
done
