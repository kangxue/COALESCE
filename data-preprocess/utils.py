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
import scipy.spatial
import scipy.ndimage

import open3d as o3d
from sklearn.neighbors import KDTree  as skKDtree


import matplotlib.cm

def expand_ones(voxel_model,  steps=1  ):

    ## to-do:  expand
    ckernel = np.ones( [steps*2+1, steps*2+1, steps*2+1], np.float32  )

    voxel_model_expanded = scipy.ndimage.filters.convolve(  voxel_model.astype(np.float32), ckernel, mode='constant' )

    voxel_model_expanded =  (   (voxel_model_expanded > 0).astype( np.uint8 )  )

    return voxel_model_expanded


def compressVoxelsFrom256To128(  voxel_model_256  ):

    voxel_model_128 = np.zeros( [128,128,128], np.uint8)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                voxel_model_128 = voxel_model_128  +  voxel_model_256[i::2,j::2,k::2]
                
    voxel_model_128 = (voxel_model_128>0).astype(np.uint8)

    return voxel_model_128


def doesPartExist( batch_points_float ):

    dims = np.max( batch_points_float, axis=0 ) - np.min(batch_points_float, axis=0)
    
    if np.linalg.norm( dims ) < 0.01:
        return False
    else:
        return True
        

def filter_yi2016_segLabels(yi2016_pts, yi2016_labels, num_of_parts ):
    

    newPts,  newLabels  = filter_yi2016_segLabels_remove(yi2016_pts, yi2016_labels, num_of_parts )

    tree = skKDtree(newPts, leaf_size=1 )
    _, indices = tree.query( yi2016_pts, k=1) 

    yi2016_labels_filtered = newLabels[ indices[:, 0] ]

    return yi2016_pts,  yi2016_labels_filtered  



def filter_yi2016_segLabels_remove(yi2016_pts, yi2016_labels, num_of_parts ):
    # assuming yi2016_pts is normalized to make the diagonal length = 1
    newPts_list = []
    newLabels_list = []  


    for lb in range(1, num_of_parts+1):
    
        partPts = yi2016_pts[ yi2016_labels==lb ]

        if(  partPts.shape[0] < 10 ):
            continue
            
        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(partPts)
        
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,  std_ratio=1.0 )  ## tuned paramters
        
        partPts_inliers = partPts[ind]
        
        newPts_list.append(  partPts_inliers)
        newLabels_list.append(  np.ones([partPts_inliers.shape[0]], dtype=np.int32 ) * lb )
     
    newPts = np.vstack(  newPts_list )
    newLabels = np.hstack(  newLabels_list )
    
    return newPts,  newLabels  


def get_onJoint(  yi2016_pts ,  yi2016_labels,  px, py, pz,  radius ):

    onJoints = False

    distances = np.linalg.norm( yi2016_pts - np.array( [px, py, pz] ) ,  axis=1 )  
    mask1 = (  distances    <  radius )

    if np.unique( yi2016_labels[mask1] ).shape[0] > 1:
        onJoints = True

    return onJoints
    


def intGridNodes_2_floatPoints( batch_points_int, voxDim ):
    batch_points = ( batch_points_int +0.5)/voxDim-0.5
    return batch_points
    


def get_onBoundary_OnJoints(  yi2016_pts ,  yi2016_labels,  px, py, pz,  radius1,  radius2):

    onBoundary = False
    onJoints = False

    distances = np.linalg.norm( yi2016_pts - np.array( [px, py, pz] ) ,  axis=1 )  
    mask1 = (  distances    <  radius1 )
    mask2 = (  distances    <  radius2 )

    if np.unique( yi2016_labels[mask1] ).shape[0] > 1:
        onJoints = True
    else:
        
        if np.unique( yi2016_labels[mask2] ).shape[0] > 1:
            onBoundary = True
            
    return onBoundary, onJoints


    
def get_JointMask_BoundaryMsk_kdtree(voxel_model,  dim_voxel,  yi2016_pts ,  yi2016_labels, erodeRd,  erodeRd_extend  ):

    tree = skKDtree(yi2016_pts, leaf_size=1 )

    jointMsk    = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)
    boundaryMsk = np.zeros([dim_voxel,dim_voxel,dim_voxel],np.uint8)

    indices = np.nonzero(voxel_model)
    ind_x = indices[0]
    ind_y = indices[1]
    ind_z = indices[2]

    nonzero_coords_int = ( np.asarray(indices) ).T
    print("nonzero_coords_int.shape ", nonzero_coords_int.shape )
    num_of_nonzeros = nonzero_coords_int.shape[0]

    nonzero_coords_float = (nonzero_coords_int + 0.5)/dim_voxel - 0.5


    NNeis_small = tree.query_radius( nonzero_coords_float, r=erodeRd, count_only=False)
    NNeis_big   = tree.query_radius( nonzero_coords_float, r=erodeRd_extend,  count_only=False)


    for t in range(num_of_nonzeros):

        if np.unique( yi2016_labels[ NNeis_small[t] ] ).shape[0] > 1:
            i = ind_x[t] 
            j = ind_y[t] 
            k = ind_z[t] 
            jointMsk[i,j,k] = 1
        else:
            
            if np.unique( yi2016_labels[ NNeis_big[t] ] ).shape[0] > 1:
                i = ind_x[t] 
                j = ind_y[t] 
                k = ind_z[t] 
                boundaryMsk[i,j,k] = 1
                
                   
    return jointMsk,  boundaryMsk





def get_segLabels( voxel_model, dim_voxel, reference_pts, reference_labels ):
    
    print("get_segLabels{ ")
    tree = skKDtree(reference_pts, leaf_size=1 )

    
    segLabels =  np.zeros([dim_voxel,dim_voxel,dim_voxel],  np.uint8) 


    indices = np.nonzero(voxel_model)
    ind_x = np.expand_dims( indices[0], axis=1)
    ind_y = np.expand_dims( indices[1], axis=1)
    ind_z = np.expand_dims( indices[2], axis=1)

    xyzs = np.concatenate( (ind_x, ind_y, ind_z), axis=1) 
    xyzs = (xyzs + 0.5 ) / dim_voxel - 0.5

    _, idx = tree.query( xyzs, k=1) 

    segLabels[indices[0], indices[1], indices[2]] = reference_labels[ idx[:, 0] ]

    print("} ")
    return segLabels  



def ConvertMask_to_samples( VoxMask,   numberLimit  ):
    indices = np.nonzero(VoxMask)
    ind_x = indices[0]
    ind_y = indices[1]
    ind_z = indices[2]
    num_of_nonzeros = len( ind_x )
    t_list =  list(range(num_of_nonzeros)) 
    random.shuffle( t_list ) 

    t_list = t_list[ : numberLimit ]  # no more than numberLimit

    sample_points_1 = np.array(  [ ind_x[ t_list ], ind_y[ t_list ],  ind_z[ t_list ] ],  np.uint8) 

    return sample_points_1.T



def sample_surface_point( voxel_model,   dim_voxel, targetNum ):


    # get surface mask
    shift_1 = np.roll(voxel_model, -1, axis=0)
    shift_2 = np.roll(voxel_model,  1, axis=0)
    shift_3 = np.roll(voxel_model, -1, axis=1)
    shift_4 = np.roll(voxel_model,  1, axis=1)
    shift_5 = np.roll(voxel_model, -1, axis=2)
    shift_6 = np.roll(voxel_model,  1, axis=2)

    shiftSum = shift_1 + shift_2 + shift_3 + shift_4 + shift_5 + shift_6 + voxel_model
    SurfaceMsk_1 = (np.logical_and( shiftSum != 0 , shiftSum != 7 )).astype(np.uint8)

    SurfaceMsk_expanded = np.copy( SurfaceMsk_1 )
    for i in range(5):
        SurfaceMsk_expanded = expand_ones( SurfaceMsk_expanded, 1 )

    SurfaceMsk_expandedOnly = ( np.logical_and(  SurfaceMsk_expanded , SurfaceMsk_1==0  ) ).astype(np.uint8)

	####################


    # add  surface samples
    sample_points_1 = ConvertMask_to_samples( SurfaceMsk_1,   targetNum  )
    # add expanded surface samples
    sample_points_2 = ConvertMask_to_samples( SurfaceMsk_expandedOnly,   targetNum  )

    # merge samples
    sample_points_12 = np.vstack( [sample_points_1, sample_points_2]  )
    count = sample_points_12.shape[0]

    sample_points = np.zeros([targetNum*10, 3],np.uint8)
    sample_points[:count, :] = sample_points_12


    print( "samples around surface: ", count)

    # add random samples
    when_to_stop = min( int(count  + count*0.11 ),  targetNum*10  )
    when_to_stop = max( when_to_stop, targetNum)
    while count< when_to_stop:

        sample_points[count, 0]  = random.randint(0, dim_voxel-1)
        sample_points[count, 1]  = random.randint(0, dim_voxel-1)
        sample_points[count, 2]  = random.randint(0, dim_voxel-1)

        count = count+1
        

    #### randomly select targetNum
    sample_points = sample_points[:count, : ]

    sample_points = sample_points[  np.random.choice( sample_points.shape[0],  targetNum), :]
    
    sample_values_shape = voxel_model[ sample_points[:,0] ,  sample_points[:,1] ,  sample_points[:,2]  ]
    sample_values_shape = np.expand_dims(sample_values_shape, axis=1) 

    
    
    return sample_points,  sample_values_shape
    




def sample_surface_point_around_joints( voxel_model,  jointMask, dim_voxel, targetNum ):


    # expand jointMask
    jointMask_expanded = jointMask
    for i in range(5):
        jointMask_expanded = expand_ones( jointMask_expanded, 1 )


    # get surface mask
    shift_1 = np.roll(voxel_model, -1, axis=0)
    shift_2 = np.roll(voxel_model,  1, axis=0)
    shift_3 = np.roll(voxel_model, -1, axis=1)
    shift_4 = np.roll(voxel_model,  1, axis=1)
    shift_5 = np.roll(voxel_model, -1, axis=2)
    shift_6 = np.roll(voxel_model,  1, axis=2)

    shiftSum = shift_1 + shift_2 + shift_3 + shift_4 + shift_5 + shift_6 + voxel_model
    SurfaceMsk_1 = (np.logical_and( shiftSum != 0 , shiftSum != 7 )).astype(np.uint8)
    


    ### sample parts around joints
    SurfaceMsk_1_joint = ( np.logical_and( SurfaceMsk_1, jointMask_expanded)  ).astype(np.uint8)
    sample_points_1 = ConvertMask_to_samples( SurfaceMsk_1_joint,   int(targetNum * 0.4 * 2)  )  # 40%

    ## sample shell
    shellMsk = ( np.logical_and( SurfaceMsk_1, (voxel_model + jointMask_expanded)==0 )  ).astype(np.uint8)
    sample_points_2 = ConvertMask_to_samples( shellMsk,   int(targetNum  * 0.10 * 2)  )   # 10%


    ### sample jointMask_expandedOnly
    jointMask_expandedOnly = ( np.logical_and( jointMask_expanded, SurfaceMsk_1_joint==0 )  ).astype(np.uint8)
    sample_points_3 = ConvertMask_to_samples( jointMask_expandedOnly,   int(targetNum  * 0.4 * 2 )  )   # 40%


    # merge samples
    sample_points_123 = np.vstack( [sample_points_1, sample_points_2,  sample_points_3 ]  )
    count = sample_points_123.shape[0]

    sample_points = np.zeros([targetNum*10, 3],np.uint8)
    sample_points[:count, :] = sample_points_123

    print( "samples around surface: ", count)

    jointSampleNum = count


    # add random samples
    when_to_stop = min( int(count  + count*0.11 ),  targetNum*10  )     # 10%
    when_to_stop = max( when_to_stop, targetNum)
    while count< when_to_stop:
        ii = random.randint(0, dim_voxel-1)
        jj = random.randint(0, dim_voxel-1)
        kk = random.randint(0, dim_voxel-1)

        if not voxel_model[ii, jj, kk]:
            sample_points[count, 0]  = ii
            sample_points[count, 1]  = jj
            sample_points[count, 2]  = kk

            count = count+1
        

    #### randomly select targetNum
    sample_points = sample_points[:count, : ]

    sample_points = sample_points[  np.random.choice( sample_points.shape[0],  targetNum), :]
    
    sample_values_shape = voxel_model[ sample_points[:,0] ,  sample_points[:,1] ,  sample_points[:,2]  ]
    sample_values_shape = np.expand_dims(sample_values_shape, axis=1) 

    sample_values_joint = sample_values_shape
    
    
    return sample_points,  sample_values_shape,   sample_values_joint, jointSampleNum



def Sample_Points_Values_on_surface_parts(  voxel_labels, voxel_model, jointMsk, dim_voxel, targetNum_per_parts, num_of_parts  ):

    sample_points_list = []
    sample_values_list = []

    print("shape of voxel_labels = ", voxel_labels.shape )


    for lb in range(1, num_of_parts+1 ):
            
        partMsk = (voxel_labels == lb).astype(np.uint8)

        
        occludedRegion = ( np.logical_and(  voxel_model , partMsk==0  ) ).astype(np.uint8)


        # get surface mask
        partMsk_shift_1 = np.roll(partMsk, -1, axis=0)
        partMsk_shift_2 = np.roll(partMsk,  1, axis=0)
        partMsk_shift_3 = np.roll(partMsk, -1, axis=1)
        partMsk_shift_4 = np.roll(partMsk,  1, axis=1)
        partMsk_shift_5 = np.roll(partMsk, -1, axis=2)
        partMsk_shift_6 = np.roll(partMsk,  1, axis=2)

        shiftSum = partMsk_shift_1 + partMsk_shift_2 + partMsk_shift_3 + partMsk_shift_4 + partMsk_shift_5 + partMsk_shift_6 + partMsk
        partSurfaceMsk_1 = (np.logical_and( shiftSum != 0 , shiftSum != 7 )).astype(np.uint8)


        partSurfaceMsk_5 = np.copy( partSurfaceMsk_1 )
        for i in range(5):
            partSurfaceMsk_5 = expand_ones( partSurfaceMsk_5, 1 )


        partSurfaceMsk_1_beforeSample = (np.logical_and( partSurfaceMsk_1 ,  (jointMsk + occludedRegion) == 0  )).astype(np.uint8)
 
        partSurfaceMsk_5 = ( np.logical_and(  partSurfaceMsk_5 ,  (partSurfaceMsk_1 + jointMsk + occludedRegion )==0  ) ).astype(np.uint8)


        # add surface samples for partSurfaceMsk_1_beforeSample
        sample_points_1 = ConvertMask_to_samples( partSurfaceMsk_1_beforeSample,   targetNum_per_parts  )   # 45 %

        # add surface samples for partSurfaceMsk_5       
        sample_points_2 = ConvertMask_to_samples( partSurfaceMsk_5,   targetNum_per_parts  )  # 45 %

        # merge samples
        sample_points_12 = np.vstack( [sample_points_1, sample_points_2]  )
        count = sample_points_12.shape[0]

        sample_points = np.zeros([targetNum_per_parts*10, 3],np.uint8)
        sample_points[:count, :] = sample_points_12

        print( "samples around surface: ", count)



        # add random samples
        when_to_stop = min( int(count  + count*0.11 ),  targetNum_per_parts*10  )    # 10%
        when_to_stop = max(when_to_stop, targetNum_per_parts)
        while count< when_to_stop:
            ii = random.randint(0, dim_voxel-1)
            jj = random.randint(0, dim_voxel-1)
            kk = random.randint(0, dim_voxel-1)

            if  not voxel_model[ii, jj, kk]:
                sample_points[count, 0]  = ii
                sample_points[count, 1]  = jj
                sample_points[count, 2]  = kk

                count = count+1


        #### randomly select targetNum_per_parts
        sample_points = sample_points[:count, : ]
        sample_points = sample_points[  np.random.choice( sample_points.shape[0],  targetNum_per_parts), :]
        sample_values = partMsk[ sample_points[:,0] ,  sample_points[:,1] ,  sample_points[:,2]  ]
        sample_values = np.expand_dims(sample_values, axis=1) 


        sample_points_list.append(  np.expand_dims(sample_points, axis=0)  )
        sample_values_list.append(  np.expand_dims(sample_values, axis=0)  )

    points_part = np.concatenate(sample_points_list, axis=0 )
    values_part = np.concatenate(sample_values_list, axis=0 )

    return  points_part, values_part




def Sample_complete_parts(  voxel_labels, voxel_model, dim_voxel, targetNum_per_parts, num_of_parts  ):

    sample_points_list = []
    sample_values_list = []

    print("shape of voxel_labels = ", voxel_labels.shape )


    for lb in range(1, num_of_parts+1 ):
            
        sample_points = np.zeros([targetNum_per_parts*10, 3],np.uint8)
        
        partMsk = (voxel_labels == lb).astype(np.uint8)
        
        occludedRegion = ( np.logical_and(  voxel_model , partMsk==0  ) ).astype(np.uint8)

        # get surface mask
        partMsk_shift_1 = np.roll(partMsk, -1, axis=0)
        partMsk_shift_2 = np.roll(partMsk,  1, axis=0)
        partMsk_shift_3 = np.roll(partMsk, -1, axis=1)
        partMsk_shift_4 = np.roll(partMsk,  1, axis=1)
        partMsk_shift_5 = np.roll(partMsk, -1, axis=2)
        partMsk_shift_6 = np.roll(partMsk,  1, axis=2)

        shiftSum = partMsk_shift_1 + partMsk_shift_2 + partMsk_shift_3 + partMsk_shift_4 + partMsk_shift_5 + partMsk_shift_6 + partMsk
        partSurfaceMsk_1 = (np.logical_and( shiftSum != 0 , shiftSum != 7 )).astype(np.uint8)
        partSurfaceMsk_1 = (np.logical_and( partSurfaceMsk_1 , occludedRegion==0 )).astype(np.uint8)


        partSurfaceMsk_5 = np.copy( partSurfaceMsk_1 )
        for i in range(5):
            partSurfaceMsk_5 = expand_ones( partSurfaceMsk_5, 1 )
        partSurfaceMsk_5 = (np.logical_and( partSurfaceMsk_5 , (partSurfaceMsk_1 + occludedRegion)==0 )).astype(np.uint8)



        # add surface samples for partSurfaceMsk_1
        sample_points_1 = ConvertMask_to_samples( partSurfaceMsk_1,   targetNum_per_parts  )

        # add surface samples for partSurfaceMsk_5       
        sample_points_2 = ConvertMask_to_samples( partSurfaceMsk_5,   targetNum_per_parts  )

        # merge samples
        sample_points_12 = np.vstack( [sample_points_1, sample_points_2]  )
        count = sample_points_12.shape[0]

        sample_points = np.zeros([targetNum_per_parts*10, 3],np.uint8)
        sample_points[:count, :] = sample_points_12

        print( "samples around surface: ", count)



        # add random samples
        when_to_stop = min( int(count  + count*0.11 ),  targetNum_per_parts*10  )
        when_to_stop = max(when_to_stop, targetNum_per_parts)
        while count< when_to_stop:
            ii = random.randint(0, dim_voxel-1)
            jj = random.randint(0, dim_voxel-1)
            kk = random.randint(0, dim_voxel-1)

            if  not voxel_model[ii, jj, kk]:
                sample_points[count, 0]  = ii
                sample_points[count, 1]  = jj
                sample_points[count, 2]  = kk

                count = count+1


        #### randomly select targetNum_per_parts
        sample_points = sample_points[:count, : ]
        sample_points = sample_points[  np.random.choice( sample_points.shape[0],  targetNum_per_parts), :]
        sample_values = partMsk[ sample_points[:,0] ,  sample_points[:,1] ,  sample_points[:,2]  ]
        sample_values = np.expand_dims(sample_values, axis=1) 


        sample_points_list.append(  np.expand_dims(sample_points, axis=0)  )
        sample_values_list.append(  np.expand_dims(sample_values, axis=0)  )

    points_part = np.concatenate(sample_points_list, axis=0 )
    values_part = np.concatenate(sample_values_list, axis=0 )

    return  points_part, values_part




def list_image(root, recursive, exts):
    image_list = []
    cat = {}
    for path, subdirs, files in os.walk(root):
        for fname in files:
            fpath = os.path.join(path, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                if path not in cat:
                    cat[path] = len(cat)
                image_list.append((os.path.relpath(fpath, root), cat[path]))
    return image_list

############################ IO

def load_ply(file_name ):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T

    return points
	

def load_ply_withNormals(file_name ):
    
    ply_data = PlyData.read(file_name)
    vertices = ply_data['vertex']
    
    points  = np.vstack([vertices['x'],  vertices['y'],  vertices['z']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    return points, normals


def output_point_cloud_ply(xyz,  filepath ):

        print('write: ' + filepath)

        with open( filepath, 'w') as f:
            pn = xyz.shape[0]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f\n' % (xyz[i][0],  xyz[i][1],  xyz[i][2]) )
				


def output_point_cloud_valued_ply(xyz,  filepath,  values ):

        print('write: ' + filepath)

        with open( filepath, 'w') as f:
            pn = xyz.shape[0]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(pn):
                CL = matplotlib.cm.hsv( values[i] )
                f.write('%f %f %f %d %d %d\n' % (xyz[i][0],  xyz[i][1],  xyz[i][2], CL[0]*255, CL[1]*255, CL[2]*255   ) )
				

def output_point_cloud_valued_ply_withNormal(filepath, xyz,  normals,   values ):

        print('write: ' + filepath)

        with open( filepath, 'w') as f:
            pn = xyz.shape[0]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn) )
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property float nx\n')
            f.write('property float ny\n')
            f.write('property float nz\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(pn):
                CL = matplotlib.cm.hsv( values[i] )
                f.write('%f %f %f %f %f %f %d %d %d\n' % (xyz[i][0],  xyz[i][1],  xyz[i][2], \
                normals[i][0],  normals[i][1],  normals[i][2], \
                 CL[0]*255, CL[1]*255, CL[2]*255   ) )
				



def write_colored_ply( filepath, xyz, normals,  labels):
    print('write: ' + filepath)

    #colors = np.zeros( xyz.shape, np.int32 )
    colorTabe = np.array( [ [255, 0, 0],  \
                            [0, 255, 0],  \
                            [0, 0, 255],   \
                            [0, 0, 0],  \
                            [255, 255, 255],   \
                            [0, 255, 255],  \
                            [255, 0, 255],   \
                            [255, 255, 0]        ] )

    colors = colorTabe[labels, :]

    if normals is not None:
        with open(filepath, 'w') as f:
            pn = xyz.shape[0]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property float nx\n')
            f.write('property float ny\n')
            f.write('property float nz\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f %f %f %f %d %d %d\n' % (xyz[i][0], xyz[i][1], xyz[i][2], normals[i][0], normals[i][1], normals[i][2], colors[i][0], colors[i][1], colors[i][2] ))
    else:
        with open(filepath, 'w') as f:
            pn = xyz.shape[0]
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n' % (pn))
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('end_header\n')
            for i in range(pn):
                f.write('%f %f %f %d %d %d\n' % (xyz[i][0], xyz[i][1], xyz[i][2],  colors[i][0], colors[i][1], colors[i][2] ))


################ part segment, and process
# segment
def getDenseSegLabels( densePoints,  yi2016_pts, yi2016_labels):
    
    
    tree = skKDtree(yi2016_pts, leaf_size=1 )
    _, indices = tree.query( densePoints, k=1) 

    denseSegLables = yi2016_labels[ indices[:, 0] ]
    return denseSegLables



# make all parts 2048
def makeEveryPart2048( densePoints,  denseSegLables ):

    uniqueLabels = np.unique( denseSegLables )

    partList = []
    labelList = []
    for lb in uniqueLabels:
        partPts = densePoints[denseSegLables==lb, : ]

        while partPts.shape[0] < 2048:
            partPts = np.concatenate(  (partPts,  partPts), axis=0 )
        np.random.shuffle( partPts )
        
        partList.append(   partPts[:2048,:]  )        
        labelList.append(  np.ones(2048,dtype=int )*lb  )
    
    densePoints_2048     = np.concatenate(   partList , axis=0 )
    denseSegLables_2048  = np.concatenate(   labelList , axis=0 )

    return densePoints_2048, denseSegLables_2048




def randomDisplaceScale_Parts( points,  labels,  displaceRange=0.025, scaleRange=0.1 ):

    DR = displaceRange
    SR = scaleRange

    uniqueLabels = np.unique( labels )

    partList = []
    for ii in range( len(uniqueLabels) ) :

        partPts = points[ii*2048 : (ii+1)*2048, : ]

        partPts = partPts +  np.array(  [ random.random() * DR*2 - DR,  random.random() *  DR*2 - DR,   random.random() *  DR*2 - DR ] )
        partPts = partPts * ( 1 + random.random() * SR*2 - SR )
        
        partList.append( partPts )

    points_transformed = np.concatenate(  partList, axis=0 )

    return points_transformed


