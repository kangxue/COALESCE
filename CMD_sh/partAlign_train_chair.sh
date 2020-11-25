#!/bin/bash

exec 2>&1 
set -x #echo on

DataDir="./03001627_Chair/03001627_sampling_erode0.05_256_ptsSorted"
wordir="03001627_Chair"
gpu_id="0"
dataset="03001627_vox"

diffShape="0"
num_of_parts="4"

erodeRadius="0.05"

echo $DataDir
echo $gpu_id
echo $dataset
echo $erodeRadius


python -u partAlign.py --train  --epoch 200  --num_of_parts $num_of_parts --gpu $gpu_id  --reWei 0 --data_dir $DataDir  --dataset $dataset   --learning_rate 0.001 --shapeBatchSize  8


if [ $diffShape -eq "0" ]; then
	echo "diffShape is 0"
else
	echo "diffShape is not 0"
fi


python -u partAlign.py          --epoch 200  --num_of_parts  $num_of_parts --gpu $gpu_id  --reWei 0 --data_dir $DataDir  --dataset $dataset  --diffShape $diffShape
