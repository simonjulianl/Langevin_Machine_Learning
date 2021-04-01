#!/bin/bash

for i in {13..13}; do
  mkdir $1/run$i;
  cd $1/run$i
  cat ../../MC_parameters.tmpl | sed s/@@/$i/g > MC_parameters.py
  cp ../../cp2parameters_folder.sh .
  cd ../../
done


