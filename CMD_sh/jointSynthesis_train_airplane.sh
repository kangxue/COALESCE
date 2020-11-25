#!/bin/bash

exec 2>&1 
set -x #echo on


DataDir="./02691156_Airplane/02691156_sampling_erode0.025_256_ptsSorted"
gpu_id="0"
dataset="02691156_vox"
num_of_parts="4"


echo $DataDir
echo $gpu_id
echo $dataset
echo $num_of_parts



python -u jointSynthesis.py --train  --epoch 20  --num_of_parts  $num_of_parts --gpu $gpu_id  -data_dir $DataDir  --dataset $dataset    --learning_rate  0.0001  --ptsBatchSize 2048 
python -u jointSynthesis.py --train  --epoch 40  --num_of_parts  $num_of_parts --gpu $gpu_id  -data_dir $DataDir  --dataset $dataset    --learning_rate  0.0001  --ptsBatchSize 4096 
python -u jointSynthesis.py --train  --epoch 60  --num_of_parts  $num_of_parts --gpu $gpu_id  -data_dir $DataDir  --dataset $dataset    --learning_rate  0.0001  --ptsBatchSize 8192 
python -u jointSynthesis.py --train  --epoch 80  --num_of_parts  $num_of_parts --gpu $gpu_id  -data_dir $DataDir  --dataset $dataset    --learning_rate  0.0001  --ptsBatchSize 16384 
python -u jointSynthesis.py --train  --epoch 160 --num_of_parts  $num_of_parts --gpu $gpu_id  -data_dir $DataDir  --dataset $dataset    --learning_rate  0.0001  --ptsBatchSize 16384

python -u jointSynthesis.py          --epoch 160  --num_of_parts  $num_of_parts --gpu $gpu_id  -data_dir $DataDir  --dataset $dataset   --outputHdf5 1 --FTSteps 0

