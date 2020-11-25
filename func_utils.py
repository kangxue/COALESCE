import numpy as np
import os
import h5py
from scipy.io import loadmat
import random
import json
import io

import argparse
import datetime

#from plyfile import  PlyData

import random
import scipy

import open3d as o3d



############################ IO

def load_ply(file_name ):

    pcd = o3d.io.read_point_cloud(  file_name  )
    points = np.asarray(pcd.points)

    return points
	

def load_ply_withnorml(file_name ):

    pcd = o3d.io.read_point_cloud(  file_name  )

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    return points, normals



def load_ply_withfaces(file_name ):
    
    if not os.path.exists(  file_name ):
        points = np.zeros( [0,3], np.float32 )
        tri_data = np.zeros( [0,3], np.int32 )
        return points, tri_data
       
    mesh = o3d.io.read_triangle_mesh(  file_name  )
    points = np.asarray(mesh.vertices )
    tri_data = np.asarray(mesh.triangles)


    return points, tri_data
    


def write_ply_withfaces(filepath, xyz, faces ):

    print( "write_ply_withfaces: " + filepath )
    
    with open(filepath, 'w', buffering=10240000) as f:
        pn = xyz.shape[0]
        fn = faces.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (pn))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %d\n' % (fn))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        for i in range(pn):
            f.write('%f %f %f\n' % (xyz[i][0], xyz[i][1], xyz[i][2] ))
        for i in range(fn):
            f.write('3 %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2] ))


def output_point_cloud_ply(filepath, xyz ):

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
        with open(filepath, 'w', buffering=10240000) as f:
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
        with open(filepath, 'w', buffering=10240000) as f:
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



def mergeMeshFiles( meshPathList, outputPath):
    
    ########
    ply_list = []
    face_list = []


    for meshPath in meshPathList:

        if os.path.exists( meshPath ):
            ply3, face3 = load_ply_withfaces( meshPath )
            ply_list.append( ply3 )
            face_list.append( face3 )


    shift_n = 0
    for i in range(1, len(face_list)):
        shift_n =  shift_n + ply_list[i-1].shape[0]
        face_list[i] = face_list[i] + shift_n
    
        
    points = np.concatenate( ply_list, axis=0 )
    faces  = np.concatenate( face_list, axis=0 )

    write_ply_withfaces( outputPath,  points,  faces )



def mergeMesh_parts( vert_List, face_List ):
    
    face_List_new = []
    face_List_new.append( face_List[0] )

    ########
    shift_n = 0
    for i in range(1, len(face_List)):
        shift_n =  shift_n + vert_List[i-1].shape[0]
        face_List_new.append( face_List[i] + shift_n )
    
        
    points = np.concatenate( vert_List, axis=0 )
    faces  = np.concatenate( face_List_new, axis=0 )

    return points, faces


def loadObjNameList( fname ):

    # load obj list
    if os.path.exists( fname  ):
        text_file = open( fname, "r")
        objlist = text_file.readlines()
            
        for i in range(len( objlist)):
            objlist[i] = objlist[i].rstrip('\r\n')
                
        text_file.close()
    else:
        print("error: cannot load "+ fname)
        exit(0)
    
    return objlist
        

def getPartInfo( work_dir):

    if work_dir=="03001627_Chair":
        objNum = 3746
        num_of_parts = 4
        PartLabels = [ [2,1], [2,3], [2,4] ]

    elif work_dir=="02691156_Airplane":
        objNum = 2690
        num_of_parts = 4
        PartLabels = [ [1,2], [1,3], [1,4] ]

    elif work_dir=="03797390_Mug":
        objNum = 184
        num_of_parts = 2
        PartLabels = [ [2,1] ]

    else:
        print( work_dir + " not matched!!!")
        exit(0)

    return objNum, num_of_parts, PartLabels
        

def getMeshInfo( work_dir, data_dir ):

    if work_dir=="03001627_Chair":
        part_name_list = ['12_back', '13_seat', '14_leg', '15_arm']
        part_mesh_dir  = '../part_mesh/shapenet_mesh_chair/'
		
    elif work_dir=="02691156_Airplane":
        part_name_list = ['1_body', '2_wing', '3_tail', '4_engine']
        part_mesh_dir  = '../part_mesh/shapenet_mesh_plane/'

    elif work_dir=="03797390_Mug":
        part_name_list = ['1_handle', '2_body' ]
        part_mesh_dir  = '../part_mesh/shapenet_mesh_mug/'

    else:
        print( work_dir + " not matched!!!")
        exit(0)

    if "0.05_" in data_dir:
        part_mesh_dir =  part_mesh_dir + 'part_mesh0050/'
    if "0.025_" in data_dir:
        part_mesh_dir =  part_mesh_dir + 'part_mesh0025/'
    if "0.0_" in data_dir:
        part_mesh_dir =  part_mesh_dir + 'part_mesh0000/'

    return part_name_list, part_mesh_dir
        


def get_random_dis_vector(DR):
    dis_vector = np.array(  [ random.random() * DR*2 - DR,  random.random() *  DR*2 - DR,   random.random() *  DR*2 - DR ] )
    return dis_vector



def get_random_scales__nonIsotropicScale(SR, isotropicScale=False ):

    assert(  isotropicScale==False )  # only isotropicScale==True is allowed


    s1 =  1 + random.random() * SR*2 - SR 
    s2 =  1 + random.random() * SR*2 - SR 
    s3 =  1 + random.random() * SR*2 - SR 
    if isotropicScale:
        scales = np.array( [s1,  s1,  s1] )
    else:
        scales = np.array( [s1,  s2,  s3] )

    return scales




def get_random_scales(SR, isotropicScale=True ):

    assert(  isotropicScale==True )  # only isotropicScale==True is allowed


    s1 =  1 + random.random() * SR*2 - SR 
    s2 =  1 + random.random() * SR*2 - SR 
    s3 =  1 + random.random() * SR*2 - SR 
    if isotropicScale:
        scales = np.array( [s1,  s1,  s1] )
    else:
        scales = np.array( [s1,  s2,  s3] )

    return scales


def normalizePointCloud( points_batch ):

    normalziedPts = np.copy(points_batch)

    if len(points_batch.shape)==2:
                
        bbox_min = np.min( normalziedPts, axis=0 )
        bbox_max = np.max( normalziedPts, axis=0 )

        normalziedPts = normalziedPts - ( bbox_max+bbox_min)/2
        normalziedPts = normalziedPts / max(  np.linalg.norm( bbox_max - bbox_min ) , 0.001  )

    else:
        assert( len(points_batch.shape)==3 )

        for i in range(points_batch.shape[0]):

            pts = points_batch[i, :, :]
                    
            bbox_min = np.min( pts, axis=0 )
            bbox_max = np.max( pts, axis=0 )

            pts = pts - ( bbox_max+bbox_min)/2
            pts = pts / max(  np.linalg.norm( bbox_max - bbox_min ) , 0.001  )

            normalziedPts[i, :, :] = pts

    return normalziedPts


def normalizePointCloud_withMeshVert( points_batch, meshVert ):

    normalziedPts = np.copy(points_batch)
    normalziedVert = np.copy(meshVert)

    if len(points_batch.shape)==2:
                
        bbox_min = np.min( normalziedPts, axis=0 )
        bbox_max = np.max( normalziedPts, axis=0 )

        normalziedPts = normalziedPts - ( bbox_max+bbox_min)/2
        normalziedPts = normalziedPts / max(   np.linalg.norm( bbox_max - bbox_min ),  0.001  )

        normalziedVert = normalziedVert - ( bbox_max+bbox_min)/2
        normalziedVert = normalziedVert / max(  np.linalg.norm( bbox_max - bbox_min ) , 0.001  )

    else:
        assert( len(points_batch.shape)==3 )

        for i in range(points_batch.shape[0]):

            bbox_min = np.min( normalziedPts[i, :, :], axis=0 )
            bbox_max = np.max( normalziedPts[i, :, :], axis=0 )

            normalziedPts[i, :, :] = normalziedPts[i, :, :] - ( bbox_max+bbox_min)/2
            normalziedPts[i, :, :] = normalziedPts[i, :, :] / max( np.linalg.norm( bbox_max - bbox_min ) , 0.001  )
                
            normalziedVert[i, :, :] = normalziedVert[i, :, :] - ( bbox_max+bbox_min)/2
            normalziedVert[i, :, :] = normalziedVert[i, :, :] / max( np.linalg.norm( bbox_max - bbox_min ) , 0.001  )
            

    return normalziedPts, normalziedVert




def randomDisplaceScale_PointSet( points___,  displaceRange=0.0, scaleRange=0.0, isotropicScale=True ):

    points = np.copy(points___)

    DR = displaceRange
    SR = scaleRange

    if len(points.shape)==2:
        
        scales = get_random_scales(SR, isotropicScale )
        dis_vector = get_random_dis_vector(DR)

        points = np.multiply( points,  scales )
        points = points +  dis_vector

    elif len(points.shape)==3:
        for i in range(points.shape[0]):
                
            scales = get_random_scales(SR, isotropicScale )
            dis_vector = get_random_dis_vector(DR)
            
            points[i] = np.multiply( points[i],  scales )
            points[i] = points[i] +  dis_vector
    else:
        print("Error,  len(points.shape)==", len(points.shape) )
        exit(0)
    
    return points



def randomDisplaceScale_TwoPointSet( points_1__,  points_2__,  displaceRange=0.0, scaleRange=0.0, isotropicScale=True ):

    points_1 = np.copy(points_1__)
    points_2 = np.copy(points_2__)

    DR = displaceRange
    SR = scaleRange

    if len(points_1.shape)==2:
        scales = get_random_scales(SR, isotropicScale )
        dis_vector = get_random_dis_vector(DR)

        points_1 = np.multiply( points_1,  scales )
        points_1 = points_1 +  dis_vector
        
        points_2 = np.multiply( points_2,  scales )
        points_2 = points_2 +  dis_vector
        
    elif len(points_1.shape)==3:
        for i in range(points_1.shape[0]):
            scales = get_random_scales(SR, isotropicScale )
            dis_vector = get_random_dis_vector(DR)

            points_1[i] = np.multiply( points_1[i] ,  scales )
            points_1[i] = points_1[i] +  dis_vector
            
            points_2[i] = np.multiply( points_2[i] ,  scales )
            points_2[i] = points_2[i] +  dis_vector
    else:
        print("Error,  len(points_1.shape)==", len(points_1.shape) )
        exit(0)
    

    return points_1, points_2





def randomDisplaceScale_PointSet_factors( points_1__,  displaceRange=0.05, scaleRange=0.0, isotropicScale=True ):


    points_1 = np.copy(points_1__)

    DR = displaceRange
    SR = scaleRange

    dis_vector = None
    scales = None
    if len(points_1.shape)==2:

        scales = get_random_scales(SR, isotropicScale )
        dis_vector = get_random_dis_vector(DR)

        points_1 = np.multiply( points_1,  scales )
        points_1 = points_1 +  dis_vector
        
    elif len(points_1.shape)==3:
        
        dis_vector_list = []
        scale_list = []
        for i in range(points_1.shape[0]):
            scales = get_random_scales(SR, isotropicScale )
            dis_vector = get_random_dis_vector(DR)

            points_1[i] = np.multiply( points_1[i] ,  scales )
            points_1[i] = points_1[i] +  dis_vector

            scale_list.append(      np.expand_dims( scales,      0 )  )
            dis_vector_list.append( np.expand_dims( dis_vector, 0 )  )

        dis_vector = np.concatenate(dis_vector_list)
        scales = np.concatenate(scale_list)

    else:
        print("Error,  len(points_1.shape)==", len(points_1.shape) )
        exit(0)
    

    return points_1,  dis_vector, scales



def intGridNodes_2_floatPoints( batch_points_int, voxDim ):
    batch_points = ( batch_points_int +0.5)/voxDim-0.5
    return batch_points
    

def floatPoints_2_uint8GridNodes ( batch_points, voxDim ):
    batch_points_int = ( ( batch_points + 0.5) * voxDim - 0.5 )

    batch_points_int[ batch_points_int < 0 ] = 0
    batch_points_int[ batch_points_int > voxDim-1 ] = voxDim-1

    return batch_points_int.astype( np.uint8 )



def doesPartExist( batch_points_float ):

    dims = np.max( batch_points_float, axis=0 ) - np.min(batch_points_float, axis=0)
    
    if np.linalg.norm( dims ) < 0.001:
        return False
    else:
        return True
    
def computeMeanPointDistance_ignore0( points_1, points_2, partMum ):

    error = 0.0
    count = 0

    for pid in range(partMum):
        partA = points_1[pid*2048:(pid+1)*2048, : ]
        partB = points_2[pid*2048:(pid+1)*2048, : ]

        if doesPartExist( partA )  and doesPartExist( partB ):
            error = error + np.sum(  np.linalg.norm(partA - partB,  axis=1 )  )
            count = count + 2048

            #print("part " + str(pid)  + ": " + str( np.sum(  np.linalg.norm(partA - partB,  axis=1 )  )/2048 ) )

    
    return error/count
    


