#!/bin/bash

for i in {0..2}; do
  print i
  mkdir run$i; 
  cd run$i
  cat ../MD_parameters.tmpl | sed s/@@/$i/g > MD_parameters.py
  cat ../MC_parameters.tmpl  > MC_parameters.py
  cd ../
done


