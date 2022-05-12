import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random
import json
import io
import shutil


from utils import *

from sklearn.neighbors import KDTree  as skKDtree

import argparse

parser = argparse.ArgumentParser()
FLAGS = parser.parse_args()


##################################
srcFolder = "03001627_sampling_erode0.05_256"
dstFolder = srcFolder + "_ptsSorted"
dataFname = "03001627_vox"
num_of_parts = 4
##################################



if  os.path.exists( dstFolder ) and os.path.isdir(dstFolder):
    shutil.rmtree(dstFolder)
shutil.copytree(srcFolder, dstFolder)  


if not os.path.exists( dstFolder+'/smoothed_label_ply' ):
    os.makedirs( dstFolder+'/smoothed_label_ply'  )


########### modify datasets

dstFilePrefix = dstFolder + '/' + dataFname

hdf5_path = dstFilePrefix + '.hdf5'
hdf5_file = h5py.File(hdf5_path, 'r+')

datasetNames =hdf5_file.keys()
print(datasetNames)



### sort point cloud

allpartPts2048_ori   = hdf5_file['allpartPts2048_ori'][:, :, :]
allpartPts2048_ori_normal   = hdf5_file['allpartPts2048_ori_normal'][:, :, :]
allpartPts2048_ori_dis      = hdf5_file['allpartPts2048_ori_dis'][:, : ]

name_num = allpartPts2048_ori.shape[0]
for mid in range(name_num):

    print( "mid = ", mid )

    for pid in range(num_of_parts):

        pc  = allpartPts2048_ori[        mid, pid*2048:(pid+1)*2048, : ]
        nml = allpartPts2048_ori_normal[ mid, pid*2048:(pid+1)*2048, : ]
        dis = allpartPts2048_ori_dis[    mid, pid*2048:(pid+1)*2048    ]

        indices = np.argsort(dis)

        allpartPts2048_ori[        mid, pid*2048:(pid+1)*2048, : ]  = pc[indices, :]
        allpartPts2048_ori_normal[ mid, pid*2048:(pid+1)*2048, : ]  = nml[indices, :]
        allpartPts2048_ori_dis[    mid, pid*2048:(pid+1)*2048    ]  = dis[indices ]


    if mid < 5:
        
        pc_list = []
        nml_list = []
        dis_list = []
        for pid in range(num_of_parts):
            pc  = allpartPts2048_ori[        mid, pid*2048:(pid*2048+512), : ]
            nml = allpartPts2048_ori_normal[ mid, pid*2048:(pid*2048+512), : ]
            dis = allpartPts2048_ori_dis[    mid, pid*2048:(pid*2048+512)    ]

            pc_list.append(  pc )
            nml_list.append(  nml )
            dis_list.append(  dis )
    
        pc = np.concatenate( pc_list,   axis=0 )
        nml = np.concatenate( nml_list, axis=0 )
        dis = np.concatenate( dis_list, axis=0 )

        outplyPath = dstFolder+'/smoothed_label_ply/'  +  str(mid) + '_first512.ply'
        output_point_cloud_valued_ply_withNormal( outplyPath,  pc, nml, dis*10  )


hdf5_file.create_dataset(  "allpartPts2048_ori_sorted",        data=allpartPts2048_ori,        compression=9 )
hdf5_file.create_dataset(  "allpartPts2048_ori_normal_sorted", data=allpartPts2048_ori_normal, compression=9 )
hdf5_file.create_dataset(  "allpartPts2048_ori_dis_sorted",    data=allpartPts2048_ori_dis,    compression=9 )


hdf5_file.close()
    
