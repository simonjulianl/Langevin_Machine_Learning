#!/bin/bash

#filename = ../init_config/n2/run

tmp="`ls ../init_config/n2/run10/nparticle2_new_nsim_rho0.1_T0.04_train_sampled.pt`"
#filename="`../init_config/n2/run`"

for i in {11..12}; do

    echo "$i"
    cd ../init_config/n2/run
    python combine2files.py $tmp $filname$i/*pt tmp

done

