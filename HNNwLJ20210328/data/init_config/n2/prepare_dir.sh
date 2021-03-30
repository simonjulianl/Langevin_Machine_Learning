#!/bin/bash

for i in {10..12}; do
  mkdir run$i; 
  cd run$i
  cat ../MC_parameters.tmpl | sed s/@@/$i/g > MC_parameters.py
  cp ../cp2parameters_folder.sh .
  cd ../
done


