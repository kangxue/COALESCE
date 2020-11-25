#!/bin/bash

exec 2>&1 
set -x #echo on




DataDir="./03797390_Mug/03797390_sampling_erode0.05_256_ptsSorted"
gpu_id="0"
dataset="03797390_vox"
echo $DataDir
echo $gpu_id
echo $dataset


###################
###################

python -u partAE.py --train  --epoch  20  --partLabel 1 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.001
python -u partAE.py --train  --epoch  40  --partLabel 1 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.0005
python -u partAE.py --train  --epoch  60  --partLabel 1 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.00025
python -u partAE.py --train  --epoch 100  --partLabel 1 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.000125
python -u partAE.py --partLabel 1 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset   --epoch 100
python -u partAE.py --partLabel 1 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset   --epoch 100  --FeedforwardTrainSet Ture





###################
###################

python -u partAE.py --train  --epoch  20  --partLabel 2 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.001
python -u partAE.py --train  --epoch  40  --partLabel 2 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.0005
python -u partAE.py --train  --epoch  60  --partLabel 2 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.00025
python -u partAE.py --train  --epoch 100  --partLabel 2 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset     --learning_rate 0.000125
python -u partAE.py --partLabel 2 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset   --epoch 100
python -u partAE.py --partLabel 2 --gpu $gpu_id  --data_dir $DataDir  --dataset $dataset   --epoch 100  --FeedforwardTrainSet Ture


