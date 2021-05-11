#!/bin/bash
# to calculate cv
# e.g. ./mc_temp.sh 4 3255 1
# $1,$2,$3 : nparticle, temperature, nsamples

for i in $(seq 0.03 0.08 0.69); do

        echo "n$1 T$i seed $2"

	CUDA_VISIBLE_DEVICES='' python MC_sampler.py ../data/gen_by_MC/n$1T${i}seed$2nsamples$3/MC_config.dict
done
