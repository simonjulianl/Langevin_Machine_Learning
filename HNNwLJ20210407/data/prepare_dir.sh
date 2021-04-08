#!/bin/bash

# $1,$2 : start and end run directory ID
# $3,$4 : e.g. init_config n2train
# $5 : MC_config or MD_config ML_config

for i in `eval echo {$1..$2}`; do
  echo "mkdir $3/$4run$i"
  echo "cd $3/$4run$i"
  mkdir $3/$4run$i;
  cd $3/$4run$i
  cat ../$5.tmpl | sed s/@@/$i/g > $5.dict
  cd ../../
done
