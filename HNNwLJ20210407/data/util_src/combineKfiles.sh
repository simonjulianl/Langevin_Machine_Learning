#!/bin/bash

# run something like this
# e.g. ./combineKfiles.sh ../init_config/n2 ../init_config/combined/newfilename.pt combine2files_sample_wise.py

data_dir=$1
newfilename=$2
compilefile=$3

echo "data dir $data_dir"
filenames=`ls $data_dir/*pt`

sorted_filename=`ls $filenames | sort -V`
echo "$sorted_filename"
n=0

for i in $sorted_filename
do
  if [ $n -eq 0 ]
  then
#    echo "cp $i ./tmp"
    cp $i ./tmp
  else
#    echo "combine 2 file for tmp $i tmp"
    python $compilefile tmp $i tmp
  fi
  n=$((n+1))
done

#echo cp tmp $newfilename
cp tmp $newfilename

