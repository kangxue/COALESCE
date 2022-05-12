import numpy as np
import cv2
import os
import h5py
from scipy.io import loadmat
import scipy
import random
import json
import io
import datetime
import shutil
from utils import *

import mcubes
import argparse

from plyfile import  PlyData
import psutil
from inspect import currentframe, getframeinfo
import gc



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
Plyfolder = outPlyfolder
voxel_input_folder = "../modelBlockedVoxels256/"+class_ID+"/"


print( "yi2016_dir = ", yi2016_dir )
print( "densePointCloud_dir = ", densePointCloud_dir )
print( "outPlyfolder = ", outPlyfolder )
print( "voxel_input_folder = ", voxel_input_folder )



outfolder = class_ID + '_sampling_erode' + str(ErodeRadius) +  '_256_' + str(FLAGS.pstart) + '_' + str(FLAGS.pend)
validationOutFolder = outfolder + "/validation_ply"

if not os.path.exists( outfolder ):
    os.makedirs( outfolder )

if not os.path.exists( validationOutFolder ):
    os.makedirs( validationOutFolder )



############################ setting
hdf5_path = outfolder + '/'+class_ID+'_vox.hdf5'
fout = open(outfolder + '/'+class_ID+'_vox_obj_list.txt','w',newline='')
fstatistics = open(outfolder + '/statistics.txt','w',newline='')



#### name list
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
    yi2016_gtLabel_path = Plyfolder + '/' + objname + '_yi2016-afterFilter.labels'
    if os.path.exists( yi2016_gtLabel_path  ):
        name_list_new.append( objname )

name_list = name_list_new
print("len(name_list) = ", len(name_list) )

### cut the list
name_list = name_list[ FLAGS.pstart: FLAGS.pend]
print("len(name_list) = ", len(name_list) )

name_num = len(name_list)


######################################################
#########  voxels and samples
hdf5_file = h5py.File(hdf5_path, 'w')
hdf5_file.create_dataset("voxels_256",             [name_num, 256, 256, 256, 1 ],  np.uint8, compression=9)
hdf5_file.create_dataset("voxels_256_seglabel",    [name_num, 256, 256, 256, 1 ],  np.uint8, compression=9)

hdf5_file.create_dataset("voxels_128",             [name_num, 128, 128, 128, 1 ],  np.uint8, compression=9)

targetSampleNum = 16384*2

hdf5_file.create_dataset("points_256",    [name_num, targetSampleNum,  3 ],  np.uint8, compression=9)
hdf5_file.create_dataset("values_256",    [name_num, targetSampleNum,  1 ],  np.uint8, compression=9)

sampleNum_per_part = 16384

if ErodeRadius > 0.001:
    
    hdf5_file.create_dataset("points_256_aroundjoint",    [name_num, targetSampleNum,  3 ],  np.uint8, compression=9)
    #hdf5_file.create_dataset("shapeValues_256_aroundjoint",    [name_num, targetSampleNum,  1 ],  np.uint8, compression=9)
    hdf5_file.create_dataset("jointValues_256_aroundjoint",    [name_num, targetSampleNum,  1 ],  np.uint8, compression=9)

        
    hdf5_file.create_dataset("voxels_256_jointmsk",    [name_num, 256, 256, 256, 1 ],  np.uint8, compression=9)
    hdf5_file.create_dataset("voxels_256_boundarymsk", [name_num, 256, 256, 256, 1 ],  np.uint8, compression=9)

    hdf5_file.create_dataset("num_of_samplesOnJoints",    [name_num, 1 ],  np.int32, compression=9)
    ### new masked datsets
    hdf5_file.create_dataset("voxels_256_seglabel_masked",    [name_num, 256, 256, 256, 1 ],  np.uint8, compression=9)

    hdf5_file.create_dataset("part_points_256_masked",    [name_num, num_of_parts,  sampleNum_per_part,  3 ],  np.uint8, compression=9)
    hdf5_file.create_dataset("part_values_256_masked",    [name_num, num_of_parts,  sampleNum_per_part,  1 ],  np.uint8, compression=9)
else:
    hdf5_file.create_dataset("part_points_256",    [name_num, num_of_parts,  sampleNum_per_part,  3 ],  np.uint8, compression=9)
    hdf5_file.create_dataset("part_values_256",    [name_num, num_of_parts,  sampleNum_per_part,  1 ],  np.uint8, compression=9)


for idx in range(name_num):
    print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(idx)
    fout.write(name_list[idx]+"\n")

    print( getframeinfo(currentframe()).filename , " line ", currentframe().f_lineno, ", psutil.virtual_memory() = ", psutil.virtual_memory()  ) 

    #get voxel models
    print("get voxel models...")
    name_list_idx = name_list[idx]
    proper_name = "shape_data/"+class_ID+"/"+name_list_idx
    try:
        voxel_model_mat = loadmat(voxel_input_folder+name_list_idx+".mat")
    except:
        print("error in loading")
        name_list_idx = name_list[idx-1]
        try:
            voxel_model_mat = loadmat(voxel_input_folder+name_list_idx+".mat")
        except:
            print("error in loading")
            name_list_idx = name_list[idx-2]
            try:
                voxel_model_mat = loadmat(voxel_input_folder+name_list_idx+".mat")
            except:
                print("error in loading")
                exit(0)

    

    # load YI2016 segmentation
    print("load YI2016 segmentation...")

    yi2016_pts_path     = Plyfolder + '/' + name_list[idx] + '_yi2016-afterFilter.ply'
    yi2016_gtLabel_path = Plyfolder + '/' + name_list[idx] + '_yi2016-afterFilter.labels'

    yi2016_pts = None
    yi2016_labels = None

    if os.path.exists( yi2016_gtLabel_path  ):
        
        yi2016_pts = load_ply( yi2016_pts_path )

        yi2016_labels = np.loadtxt(yi2016_gtLabel_path, dtype=np.dtype(int))

    else:
        print("error: cannot load "+ yi2016_gtLabel_path)
        exit(0)

    

    ##################### extract voxel model with resoltuion 256
    print("extract voxel model with resoltuion 256...")
    dim_voxel = 256
    voxel_model_b = voxel_model_mat['b'][:].astype(np.int32)
    voxel_model_bi = voxel_model_mat['bi'][:].astype(np.int32)-1
    voxel_model_256 = np.zeros([256,256,256],np.uint8)
    for i in range(16):
        for j in range(16):
            for k in range(16):
                voxel_model_256[i*16:i*16+16,j*16:j*16+16,k*16:k*16+16] = voxel_model_b[voxel_model_bi[i,j,k]]
    #add flip&transpose to convert coord from shapenet_v1 to shapenet_v2
    voxel_model_256 = np.flip(np.transpose(voxel_model_256, (2,1,0)),2)
    hdf5_file["voxels_256"][idx, :, :, :, :]         = np.reshape(voxel_model_256, (256,256,256,1))

    segLabel_256 = get_segLabels( voxel_model_256, dim_voxel, yi2016_pts, yi2016_labels )
    hdf5_file["voxels_256_seglabel"][idx,:,:,:,:]    = np.reshape(segLabel_256, (256, 256, 256,1))
    
    ######## compress 256 models into 128 models
    print("extract voxel model with resoltuion 128...")
    dim_voxel = 128
    voxel_model_128 = np.zeros([dim_voxel,dim_voxel,dim_voxel], np.uint8)
    multiplier = int(256/dim_voxel)
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_128[i,j,k] = np.max(voxel_model_256[i*multiplier:(i+1)*multiplier, \
                                                            j*multiplier:(j+1)*multiplier, \
                                                            k*multiplier:(k+1)*multiplier]  )
    hdf5_file["voxels_128"][idx,:,:,:,:]             = np.reshape(voxel_model_128,               (dim_voxel,dim_voxel,dim_voxel,1))


    #### compelte shape samples
    points_256,  values_256 = sample_surface_point( voxel_model_256,  256, targetSampleNum )
    hdf5_file["points_256"][idx,:,:]  = points_256
    hdf5_file["values_256"][idx,:,:]  = values_256


    if ErodeRadius > 0.001:

        dim_voxel = 256
        #  if a voxel has more than one part label in a neighbour ball,  set its  value  as 0
        voxel_model_256_jointmsk,  voxel_model_256_boundarymsk = get_JointMask_BoundaryMsk_kdtree( \
                                                        voxel_model_256, dim_voxel, yi2016_pts, yi2016_labels, ErodeRadius, ErodeRadius_extend )

        segLabel_256_masked = (  np.multiply( segLabel_256,   1-voxel_model_256_jointmsk  ) ).astype( np.uint8 )

        hdf5_file["voxels_256_jointmsk"][idx,:,:,:,:]    = np.reshape(voxel_model_256_jointmsk,      (dim_voxel,dim_voxel,dim_voxel,1))
        hdf5_file["voxels_256_boundarymsk"][idx,:,:,:,:] = np.reshape(voxel_model_256_boundarymsk,   (dim_voxel,dim_voxel,dim_voxel,1))
        hdf5_file["voxels_256_seglabel_masked"][idx,:,:,:,:] = np.reshape( segLabel_256_masked,      (dim_voxel,dim_voxel,dim_voxel,1))

    
        ############ sample point around joints
        print( "sample_surface_point_around_joints...")
        jointMask_256 = voxel_model_256_jointmsk #+ voxel_model_256_boundarymsk
        points_256_aroundjoint,  shapeValues_256_aroundjoint,  jointValues_256_aroundjoint,  jntsampleNum = sample_surface_point_around_joints( \
                                                                                        voxel_model_256,  jointMask_256, 256, targetSampleNum )

        hdf5_file["points_256_aroundjoint"][idx,:,:]  = points_256_aroundjoint
        hdf5_file["jointValues_256_aroundjoint"][idx,:,:]  = jointValues_256_aroundjoint

        hdf5_file["num_of_samplesOnJoints"][idx, 0] = jntsampleNum

        # sampling points for parts
        print( "Sample_Points_Values_on_surface_parts...")
        sample_points_part_masked, sample_values_part_masked =  Sample_Points_Values_on_surface_parts( \
                                                    segLabel_256_masked, voxel_model_256 , voxel_model_256_jointmsk,  256, sampleNum_per_part, num_of_parts  )

        hdf5_file["part_points_256_masked"][idx,:,:, :]  = sample_points_part_masked
        hdf5_file["part_values_256_masked"][idx,:,:, :]  = np.reshape( sample_values_part_masked, (num_of_parts, sampleNum_per_part,1) )
    
    else:

        sample_points_part, sample_values_part =  Sample_complete_parts( \
                                                    segLabel_256, voxel_model_256 , 256, sampleNum_per_part, num_of_parts )

        hdf5_file["part_points_256"][idx,:,:, :]  = sample_points_part
        hdf5_file["part_values_256"][idx,:,:, :]  = np.reshape( sample_values_part, (num_of_parts, sampleNum_per_part,1) )


    
    voxel_model_256_jointmsk = None
    voxels_256_boundarymsk = None
    segLabel_256 = None
    gc.collect()

    ############




######################################################
#########  point cloud
allpartPts2048_ori_array =        np.zeros(  [name_num, num_of_parts*2048,  3],  np.float32)
allpartPts2048_ori_normal_array = np.zeros(  [name_num, num_of_parts*2048,  3],  np.float32)
allpartPts2048_ori_dis_array =    np.zeros(  [name_num, num_of_parts*2048    ],  np.float32)


for idx in range(name_num):

    print(idx)

    dense_labels = np.loadtxt(Plyfolder + '/' + name_list[idx] +  '_ori.labels', dtype=np.uint8 )
    dense_dis2joint = np.loadtxt(Plyfolder + '/' + name_list[idx] +  '_ori.dis2joint', dtype=np.float32 )
    dense_points_ori, dense_normals_ori = load_ply_withNormals( Plyfolder + '/' + name_list[idx] +  '_ori.ply' )

    for lb in range(1,num_of_parts+1):     # part labels starting form 1, ...

        partPts = dense_points_ori[dense_labels==lb, : ]
        partNormals = dense_normals_ori[dense_labels==lb, : ]
        partPts_dis = dense_dis2joint[dense_labels==lb]

        if partPts.shape[0] == 0:
            partPts = np.zeros( (2048,3), np.float32 )
            partNormals = np.ones( (2048,3), np.float32 )
            partPts_dis = np.zeros( (2048), np.float32 )

        if not doesPartExist( partPts ):
            partNormals = np.ones( (2048,3), np.float32 )
            partPts_dis = np.zeros( (2048), np.float32 )

        assert( partPts.shape[0] == 2048 )
        assert( partNormals.shape[0] == 2048 )
        assert( partPts_dis.shape[0] == 2048 )

        allpartPts2048_ori_array[       idx, (lb-1)*2048:lb*2048, : ]  =  partPts
        allpartPts2048_ori_normal_array[idx, (lb-1)*2048:lb*2048, : ]  =  partNormals
        allpartPts2048_ori_dis_array[   idx, (lb-1)*2048:lb*2048   ]  = partPts_dis

hdf5_file.create_dataset(  "allpartPts2048_ori",        data=allpartPts2048_ori_array,        compression=9 )
hdf5_file.create_dataset(  "allpartPts2048_ori_normal", data=allpartPts2048_ori_normal_array, compression=9 )
hdf5_file.create_dataset(  "allpartPts2048_ori_dis", data=allpartPts2048_ori_dis_array, compression=9 )

print( "load dense point cloud of part done!! ")


####################################################
### output visualization
allpartPts2048_labels = np.ones(  2048*num_of_parts, dtype=np.uint8 )
allpartPts8192_labels = np.ones(  8192*num_of_parts, dtype=np.uint8 )
for lb in range(1, num_of_parts+1):
    allpartPts2048_labels[(lb-1)*2048 : lb*2048 ] = lb
    allpartPts8192_labels[(lb-1)*8192 : lb*8192 ] = lb

for idx in range(  min(name_num, 5) ):


    vertices, triangles = mcubes.marching_cubes( np.squeeze(hdf5_file["voxels_256"][idx])*1.0 , 0.5 )
    vertices =  intGridNodes_2_floatPoints(vertices, 256 )
    mcubes.export_mesh(vertices, triangles, validationOutFolder + "/" + str(idx)+ "_voxels_256.dae", str(idx))


    vertices, triangles = mcubes.marching_cubes( np.squeeze(hdf5_file["voxels_256_seglabel"][idx]==1)*1.0 , 0.5 )
    vertices =  intGridNodes_2_floatPoints(vertices, 256 )
    mcubes.export_mesh(vertices, triangles, validationOutFolder + "/" + str(idx)+ "_256_seglabel_1.dae", str(idx))
    
    
    vertices, triangles = mcubes.marching_cubes( np.squeeze(hdf5_file["voxels_128"][idx])*1.0 , 0.5 )
    vertices =  intGridNodes_2_floatPoints(vertices, 128 )
    mcubes.export_mesh(vertices, triangles, validationOutFolder + "/" + str(idx)+ "_voxels_128.dae", str(idx))


    
    write_colored_ply( validationOutFolder + "/" + str(idx)+ "_values_256" +  ".ply",         \
        (hdf5_file[ "points_256"][idx] +0.5)/256-0.5, None, np.squeeze( hdf5_file[ "values_256" ][idx] )  )




    if ErodeRadius > 0.001:

    
        vertices, triangles = mcubes.marching_cubes( np.squeeze(hdf5_file["voxels_256_seglabel_masked"][idx]==1)*1.0 , 0.5 )
        vertices =  intGridNodes_2_floatPoints(vertices, 256 )
        mcubes.export_mesh(vertices, triangles, validationOutFolder + "/" + str(idx)+ "_256_seglabel_masked_1.dae", str(idx))



        vertices, triangles = mcubes.marching_cubes( np.squeeze(hdf5_file["voxels_256_jointmsk"][idx])*1.0 , 0.5 )
        vertices =  intGridNodes_2_floatPoints(vertices, 256 )
        mcubes.export_mesh(vertices, triangles, validationOutFolder + "/" + str(idx)+ "_voxels_256_jointmsk.dae", str(idx))

        vertices, triangles = mcubes.marching_cubes( np.squeeze(hdf5_file["voxels_256_boundarymsk"][idx])*1.0 , 0.5 )
        vertices =  intGridNodes_2_floatPoints(vertices, 256 )
        mcubes.export_mesh(vertices, triangles, validationOutFolder + "/" + str(idx)+ "_voxels_256_boundarymsk.dae", str(idx))

        write_colored_ply( validationOutFolder + "/" + str(idx)+ "_jointValues_256_aroundjoint" +  ".ply",         \
            (hdf5_file[ "points_256_aroundjoint"][idx] +0.5)/256-0.5, None, np.squeeze( hdf5_file[ "jointValues_256_aroundjoint" ][idx] )  )



        dim_voxel = 256
        dstr = str(dim_voxel)

        part_points =   ( hdf5_file[ "part_points_"+dstr+"_masked"][idx]+0.5)/dim_voxel-0.5
        part_values        = np.squeeze(  hdf5_file[ "part_values_"+dstr+"_masked"][idx] )
        for lb in range(1, num_of_parts+1):
            write_colored_ply( validationOutFolder + "/" + str(idx)+ "_part_values_" + dstr+ '_masked_'+str(lb)+  ".ply",         part_points[lb-1], None, part_values[lb-1] )

    else:
        
        dim_voxel = 256
        dstr = str(dim_voxel)
        
        part_points =   ( hdf5_file[ "part_points_"+dstr ][idx]+0.5)/dim_voxel-0.5
        part_values        = np.squeeze(  hdf5_file[ "part_values_"+dstr ][idx] )
        for lb in range(1, num_of_parts+1):
            write_colored_ply( validationOutFolder + "/" + str(idx)+ "_part_values_" + dstr+ '_'+str(lb)+  ".ply",         part_points[lb-1], None, part_values[lb-1] )




    write_colored_ply( validationOutFolder + "/" + str(idx)+ '_ori.ply',  hdf5_file[ "allpartPts2048_ori" ][idx], hdf5_file[ "allpartPts2048_ori_normal" ][idx],  allpartPts2048_labels  )

####################################################

fout.close()
fstatistics.close()
hdf5_file.close()
print("finished")
