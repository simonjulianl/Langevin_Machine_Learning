#!/bin/bash

for i in {12..14}; do
  mkdir run$i; 
  cd run$i
  cat ../MD_parameters.tmpl | sed s/@@/$i/g > MD_parameters.py
  cp ../cp2parameters_folder.sh .
  cd ../
done


