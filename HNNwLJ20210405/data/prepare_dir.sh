#!/bin/bash

# $1,$2 : start and end run directory ID
# $3,$4 : folder name
# $5 : MC_parameters or MD_parameters 

for i in `eval echo {$1..$2}`; do
  echo "$i"
  mkdir $3/$4/run$i;
  cd $3/$4/run$i
  cat ../../$5.tmpl | sed s/@@/$i/g > $5.py
  cd ../../../
done


