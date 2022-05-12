import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import random
import json
import io

import argparse
import datetime

from plyfile import  PlyData

import random
import scipy

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--pstart',  type=int, default=0)
parser.add_argument('--pend',  type=int,   default=10000)

parser.add_argument('--class_ID',  type=str, default='03001627')
parser.add_argument('--num_of_parts',  type=int, default=4)
parser.add_argument('--erode_radius',  type=float,   default=0.05)


FLAGS = parser.parse_args()


ErodeRadius = FLAGS.erode_radius
ErodeRadius_extend = ErodeRadius+0.025

print( "ErodeRadius = ", ErodeRadius )
print( "ErodeRadius_extend = ", ErodeRadius_extend )



#class number
class_ID = FLAGS.class_ID
num_of_parts = FLAGS.num_of_parts

print(class_ID)
print(num_of_parts)


yi2016_dir = "../yi2016/" +class_ID

densePointCloud_dir = '../densePointCloud/' + class_ID
# create output folder
outPlyfolder =  class_ID + '_parts2048_ply_eroded' + str(ErodeRadius) + '_withnormal'

if not os.path.exists( outPlyfolder ):
    os.makedirs( outPlyfolder )

voxel_input_folder = "../modelBlockedVoxels256/"+class_ID+"/"


print( "yi2016_dir = ", yi2016_dir )
print( "densePointCloud_dir = ", densePointCloud_dir )
print( "outPlyfolder = ", outPlyfolder )
print( "voxel_input_folder = ", voxel_input_folder )





# segment
def getDenseSegLabels_eroded( densePoints,  yi2016_pts, yi2016_labels, radius):

    
    yi2016_labels_eroded = np.copy( yi2016_labels )
    
    for i in range(   yi2016_pts.shape[0]  ):
        px = yi2016_pts[i, 0]
        py = yi2016_pts[i, 1]
        pz = yi2016_pts[i, 2]
        if get_onJoint( yi2016_pts, yi2016_labels, px,py,pz, radius ):
            yi2016_labels_eroded[i] = 0

    tree = skKDtree(yi2016_pts, leaf_size=1 )
    _, indices = tree.query( densePoints, k=1) 

    denseSegLables = yi2016_labels_eroded[ indices[:, 0] ]

    
    return denseSegLables, yi2016_labels_eroded





# make all parts 2048
def makeEveryPart2048( densePoints, densePointsNormals, denseSegLables, num_of_parts ):


    jointPts = densePoints[denseSegLables==0, : ]

    partList = []
    normalList = []
    labelList = []
    for lb in range(1, num_of_parts+1):


        partPts = densePoints[denseSegLables==lb, : ]
        partNormal = densePointsNormals[denseSegLables==lb, : ]

        if partPts.shape[0] == 0:
            partPts    = np.zeros( (2048,3), np.float32 )
            partNormal = np.ones( (2048,3), np.float32 )

        while partPts.shape[0] < 2048:
            partPts = np.concatenate(  (partPts,  partPts), axis=0 )
            partNormal = np.concatenate(  (partNormal,  partNormal), axis=0 )

        idx = list( range( partPts.shape[0] ) )
        random.shuffle(idx)

        partPts = partPts[idx]
        partNormal = partNormal[idx]

        partList.append(   partPts[:2048,:]  )       
        normalList.append(   partNormal[:2048,:]  )        
        labelList.append(  np.ones(2048,dtype=int )*lb  )
    
    densePoints_2048     = np.concatenate(   partList , axis=0 )
    denseNormals_2048     = np.concatenate(   normalList , axis=0 )
    denseSegLables_2048  = np.concatenate(   labelList , axis=0 )


    # compute distance to joints

    if jointPts.shape[0] > 1:

        tree = skKDtree(jointPts, leaf_size=1 )
        distanceToJoint_2048, indices = tree.query( densePoints_2048, k=1) 
        distanceToJoint_2048 = distanceToJoint_2048[:,0]

    else:
        # Joints doesn't inclue points. COmpute distance to other parts
        Distance_list = []
        for pid in range( num_of_parts ):
            thisPart = partList[ pid ] 
            otherparts = np.vstack( partList[ :pid ] + partList[pid+1:] )

            print(  "thisPart.shape = ", thisPart.shape )
            print(  "otherparts.shape = ", otherparts.shape )
                
            tree = skKDtree(otherparts, leaf_size=1 )
            distances, _ = tree.query( thisPart, k=1) 
            distances = distances[:,0]
            
            Distance_list.append( distances  )
            print(  "distances.shape = ", distances.shape )

        distanceToJoint_2048  =  np.concatenate( Distance_list )
        print(  "distanceToJoint_2048.shape = ", distanceToJoint_2048.shape )


    return densePoints_2048, denseNormals_2048,  denseSegLables_2048, distanceToJoint_2048



############################ get  name list ############################

name_list = []
image_list = list_image(voxel_input_folder, True, ['.mat'])
for i in range(len(image_list)):
    imagine=image_list[i][0]
    name_list.append(imagine[0:-4])
name_list = sorted(name_list)
print("len(name_list) = ", len(name_list) )

########   only consider files that exist in Yi2016
name_list_new = []
for objname in name_list:
    yi2016_gtLabel_path = yi2016_dir + '/expert_verified/points_label/' + objname + '.seg'
    if os.path.exists( yi2016_gtLabel_path  ):
        name_list_new.append( objname )

name_list = name_list_new
print("len(name_list) = ", len(name_list) )


for ii in range( len( name_list )  ):

    if ii < FLAGS.pstart or ii >= FLAGS.pend:
        continue

    print(ii)

    objname = name_list[ii]
    yi2016_gtLabel_path = yi2016_dir + '/expert_verified/points_label/' + objname + '.seg'
    yi2016_pts_path = yi2016_dir + '/points/' + objname + '.pts'
    densePointCloud_ply_path = densePointCloud_dir + '/'+ objname + '.reoriented.16384.ply'

    if not os.path.exists( yi2016_gtLabel_path  ):
        print( yi2016_gtLabel_path, " does not exist!" )
        exit(0)

    if not os.path.exists( yi2016_pts_path  ):
        print( yi2016_pts_path, " does not exist!" )
        exit(0)
    
    if not os.path.exists( densePointCloud_ply_path  ):
        print( densePointCloud_ply_path, " does not exist!" )
        exit(0)
    

    print( "objname = ", objname )


    # load yi2016 pts and seg

    yi2016_pts = None
    yi2016_labels = None

    if os.path.exists( yi2016_gtLabel_path  ):
        pts_ori = np.loadtxt( yi2016_pts_path )
        yi2016_pts = np.concatenate( (pts_ori[:,2:3],  pts_ori[:,1:2], -pts_ori[:,0:1] ), axis=1 )   ###  swaped x and z, then flipped z
        yi2016_labels = np.loadtxt(yi2016_gtLabel_path, dtype=np.dtype(int))
    else:
        print("error: cannot load "+ yi2016_gtLabel_path)
        exit()
    
    ####################  outlier filtering   #########################
    yi2016_pts_beforeFilter = np.copy( yi2016_pts )
    yi2016_labels_beforeFilter = np.copy( yi2016_labels )

    write_colored_ply(  outPlyfolder + '/' + objname +  '_yi2016-beforeFilter.ply',   yi2016_pts, None,  yi2016_labels )
    with open( outPlyfolder + '/' + objname +  '_yi2016-beforeFilter.labels',  "w") as f:
        for id in range(  yi2016_labels.shape[0] ):
            f.write(  str( yi2016_labels[id] ) + '\n')


    yi2016_pts, yi2016_labels = filter_yi2016_segLabels(yi2016_pts, yi2016_labels, num_of_parts )

    write_colored_ply(  outPlyfolder + '/' + objname +  '_yi2016-afterFilter.ply',   yi2016_pts, None,  yi2016_labels )
    with open( outPlyfolder + '/' + objname +  '_yi2016-afterFilter.labels',  "w") as f:
        for id in range(  yi2016_labels.shape[0] ):
            f.write(  str( yi2016_labels[id] ) + '\n')



    ################################################################################
    

    # load dense points ply and normalize
    densePoints, densePointsNormals = load_ply_withNormals( densePointCloud_ply_path )
    
    bbox_min = np.min( densePoints, axis=0 )
    bbox_max = np.max( densePoints, axis=0 )

    densePoints = densePoints - ( bbox_max+bbox_min)/2
    densePoints = densePoints / np.linalg.norm( bbox_max - bbox_min )


    # segment
    if ErodeRadius > 0.001:
        denseSegLables, yi2016_labels_eroded = getDenseSegLabels_eroded( densePoints,  yi2016_pts, yi2016_labels, ErodeRadius)

        ################## if yi2016_labels_eroded doesn't contain joint points,  then recover the data to beforefilter
        if np.sum( yi2016_labels_eroded==0 ) == 0:
            yi2016_pts = yi2016_pts_beforeFilter
            yi2016_labels = yi2016_labels_beforeFilter
            denseSegLables, yi2016_labels_eroded = getDenseSegLabels_eroded( densePoints,  yi2016_pts, yi2016_labels, ErodeRadius)
    else:
        denseSegLables = getDenseSegLabels( densePoints,  yi2016_pts, yi2016_labels )
        yi2016_labels_eroded = yi2016_labels


    write_colored_ply(  outPlyfolder + '/' + objname +  '_yi2016-afterFilter-eroded.ply',   yi2016_pts, None,  yi2016_labels_eroded )
    with open( outPlyfolder + '/' + objname +  '_yi2016-afterFilter-eroded.labels',  "w") as f:
        for id in range(  yi2016_labels_eroded.shape[0] ):
            f.write(  str( yi2016_labels_eroded[id] ) + '\n')

            
    write_colored_ply(  outPlyfolder + '/' + objname +  '_densePoint.ply',   densePoints, None,  denseSegLables )
    with open( outPlyfolder + '/' + objname +  '_densePoint.labels',  "w") as f:
        for id in range(  denseSegLables.shape[0] ):
            f.write(  str( denseSegLables[id] ) + '\n')

    ##################


    # make all parts 2048
    densePoints_2048, denseNormals_2048, densePartLabels_2048, distanceToJoint_2048 = makeEveryPart2048( densePoints, densePointsNormals, denseSegLables, num_of_parts )
    
    write_colored_ply(  outPlyfolder + '/' + objname +  '_ori.ply',   densePoints_2048,  denseNormals_2048,  densePartLabels_2048 )

    with open( outPlyfolder + '/' + objname +  '_ori.labels',  "w") as f:
        for id in range(  densePartLabels_2048.shape[0] ):
            f.write(  str( densePartLabels_2048[id] ) + '\n')
            
    with open( outPlyfolder + '/' + objname +  '_ori.dis2joint',  "w") as f:
        for id in range(  distanceToJoint_2048.shape[0] ):
            f.write(  str( distanceToJoint_2048[id] ) + '\n')

