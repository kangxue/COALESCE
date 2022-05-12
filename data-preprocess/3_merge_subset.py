import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random
import json
import io

import argparse

parser = argparse.ArgumentParser()
FLAGS = parser.parse_args()

src_folders = [\
"03001627_sampling_erode0.025_256_0_500", \
"03001627_sampling_erode0.025_256_500_1000",  \
"03001627_sampling_erode0.025_256_1000_1500",  \
"03001627_sampling_erode0.025_256_1500_2000",  \
"03001627_sampling_erode0.025_256_2000_2500",  \
"03001627_sampling_erode0.025_256_2500_3000",  \
"03001627_sampling_erode0.025_256_3000_3500",  \
"03001627_sampling_erode0.025_256_3500_4000"  \
]

dstFolder = "03001627_sampling_erode0.025_256"
dataFname = "03001627_vox"



dstFilePrefix = dstFolder + '/' + dataFname


if not os.path.exists( dstFolder ):
    os.makedirs( dstFolder )

    
########## get list of Objs

objlist_merged = []
sub_counts = []

for srcFolder in src_folders:
    objlist_path = srcFolder + '/' + dataFname +'_obj_list.txt'
    
    if os.path.exists( objlist_path  ):
        text_file = open( objlist_path, "r")
        objlist = text_file.readlines()

        for i in range(len(objlist)):
            objlist[i] = objlist[i].rstrip('\r\n')
            objlist_merged.append(  objlist[i]  )

        sub_counts.append(  len( objlist )  )
                
        text_file.close()
    else:
        print("error: cannot load "+ objlist_path)
        exit(0)

name_num = len( objlist_merged )

print("sub_counts = ", sub_counts)
print("name_num = ", name_num)

assert( sum(sub_counts) ==  name_num )

########### write list of objs
fout = open(  dstFilePrefix +'_obj_list.txt', 'w',  newline='')
for name_list_idx in  objlist_merged:
    fout.write(name_list_idx+"\n")
fout.close()


########### create datasets

hdf5_path = dstFilePrefix + '.hdf5'
hdf5_file = h5py.File(hdf5_path, 'w')

srcpath_0 =  src_folders[0] + '/' + dataFname +'.hdf5'
datasetNames = []
with h5py.File(  srcpath_0, 'r') as data_dict_0:
    datasetNames = list( data_dict_0.keys() )

    for dname in datasetNames:

        dataShape = list( data_dict_0[dname].shape )
        dataShape[0] = name_num

        hdf5_file.create_dataset( dname,  dataShape ,  data_dict_0[dname].dtype, compression=4)


print(datasetNames)

##################  start merging datasets
sp = 0
for ii in range(len(src_folders)):
    
    srcFolder = src_folders[ii]
    dataCount = sub_counts[ii]

    print("srcFolder = ", srcFolder)
    print("dataCount = ", dataCount)

    srcpath = srcFolder + '/' + dataFname +'.hdf5'
    data_dict = h5py.File(  srcpath, 'r')
    

    for dname in datasetNames:
        hdf5_file[ dname ][sp : sp + dataCount  ] = data_dict[ dname ][:dataCount] 
        print( '----------- ', dname, ':  ',  data_dict[ dname ][:dataCount].shape ) 

    sp = sp + dataCount

    print("sp = ", sp)

hdf5_file.close()
    


