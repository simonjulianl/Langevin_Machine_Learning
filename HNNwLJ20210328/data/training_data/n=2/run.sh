#!/bin/bash

for i in {10..12}; do 
  mkdir run$i; 
  cd run$i
  cat ../MD_parameters.tmpl | sed s/@@/$i/g > MD_parameters.py
  cd ../
done


