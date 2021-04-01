#!/bin/bash

for i in {12..12}; do
  mkdir $1/run$i;
  cd $1/run$i
  cat ../../MD_parameters.tmpl | sed s/@@/$i/g > MD_parameters.py
  cp ../../cp2parameters_folder.sh .
  cd ../../
done


