#!/bin/bash

exec 2>&1 
set -x #echo on


DataDir="./03797390_Mug/03797390_sampling_erode0.05_256_ptsSorted"
wordir="03797390_Mug"
gpu_id="0"
dataset="03797390_vox"

num_of_parts="2"

folderDiff="03797390_Mug/test_res_partAlign_Epoch200_diffshapes"

echo $DataDir
echo $gpu_id
echo $wordir
echo $num_of_parts


python -u jointSynthesis.py  --epoch 240  --num_of_parts $num_of_parts --gpu $gpu_id  --data_dir $DataDir   --dataset $dataset  --test_input_dir $folderDiff --testStartID 0  --testEndID 1000 

