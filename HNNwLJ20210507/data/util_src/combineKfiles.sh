#!/bin/bash

# run something like this
# e.g. ./combineKfiles.sh ../gen_by_MC/train ../gen_by_MC/train/newfilename.pt combine2files_sample_wise.py n4

# HK add argv[4] train file generate files for different particles so that give nparticle

data_dir=$1
newfilename=$2
compilefile=$3

echo "=== data dir ==="
echo "$data_dir"
filenames=`ls $data_dir/$4*pt`

sorted_filename=`ls $filenames | sort -V`
echo "=== data sorted ==="
echo "$sorted_filename"
n=0

for i in $sorted_filename
do
  if [ $n -eq 0 ]
  then
#    echo "cp $i ./tmp"
    cp $i ./tmp
  else
    #echo "combine 2 file for tmp $i tmp"
    python $compilefile tmp $i tmp
  fi
  n=$((n+1))
done

#echo cp tmp $newfilename
cp tmp $newfilename

