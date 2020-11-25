#!/bin/bash

exec 2>&1 
set -x #echo on





DataDir="./03001627_Chair/03001627_sampling_erode0.05_256_ptsSorted"
wordir="03001627_Chair"
gpu_id="0"
dataset="03001627_vox"
num_of_parts="4"

folderDiff="03001627_Chair/test_res_partAlign_Epoch200_diffshapes"

echo $DataDir
echo $gpu_id
echo $wordir
echo $diffShape
echo $num_of_parts


python -u jointSynthesis.py  --epoch 160  --num_of_parts $num_of_parts --gpu $gpu_id  --data_dir $DataDir   --dataset $dataset  --test_input_dir $folderDiff --testStartID 0  --testEndID 1000 


