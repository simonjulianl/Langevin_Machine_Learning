#!/bin/bash

# e.g. ./mc_run.sh 4 
# $1: nparticle

temp=(0.03 0.11 0.19 0.27 0.35 0.43 0.51 0.59 0.67)
nsamples=(4 4 4 8 8 8 12 12 12)
for i in 3251 5928 1371 4729 7483 8976; do

        for j in $(seq 0 8); do

                echo "n$1 T${temp[$j]} seed $i nsamples ${nsamples[$j]} "

                #usage <programe> <nparticle> <temperature> <seed> <nsamples> <dq>
                CUDA_VISIBLE_DEVICES='' python  MC_sampler.py ../data/gen_by_MC/n$1T${temp[$j]}seed${i}nsamples${nsamples[$j]}/MC_config.dict
        done
done

