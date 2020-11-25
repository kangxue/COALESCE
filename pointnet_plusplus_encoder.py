'''
Created on February 4, 2017

@author: optas

'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout


import os
import sys
import collections

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR + "/pointnet_plusplus/utils")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/3d_interpolation")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/grouping")
sys.path.append(BASE_DIR + "/pointnet_plusplus/tf_ops/sampling")
from pointnet_util import pointnet_sa_module, pointnet_fp_module


def getEncoder_oneBranch_local(partPoints_input, dis2Joint, NumOfPts,  is_training, reuse,  scope_predix="part_1", verbose=True,bn_decay=None):

    scname = scope_predix+"_pointNetPP_encoder"
    with tf.variable_scope( scname  ) as sc:
        if reuse:
            sc.reuse_variables()

        is_training = tf.constant(is_training, dtype=tf.bool)


        l0_xyz = partPoints_input
        l0_points = None


        print( "partPoints_input.shape = ", partPoints_input.shape )
        if partPoints_input.shape[2] == 6:
            l0_xyz    = partPoints_input[:,:, 0:3]
            l0_points = partPoints_input[:,:, 3:6]

        
    
        # assume self.shapeBatchSize==1
        inputDims = tf.reduce_max(l0_xyz , axis=[0, 1] ) - tf.reduce_min( l0_xyz , axis=[0, 1] )
        print(  "inputDims.shape = ",inputDims.shape  )
        does_part_exist =  tf.dtypes.cast( tf.reduce_mean( tf.abs(inputDims) ) > 0.01,  tf.float32 )

        
        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=256, radius=0.05, nsample=128,
                                                            mlp=[32, 32, 64], mlp2=None, group_all=False,
                                                            is_training=is_training, bn_decay=bn_decay, scope='layer1', bn=False)

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.1, nsample=128,
                                                            mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                            is_training=is_training, bn_decay=bn_decay, scope='layer2', bn=False)

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                                            mlp=[128, 128, 128], mlp2=None, group_all=True,
                                                            is_training=is_training, bn_decay=bn_decay, scope='layerxxx', bn=False)


        output_4 = tf.reshape(l4_points, [l0_xyz.shape[0], 128] )


        print(('output_4.shape = %s', output_4.shape))

        return output_4 * does_part_exist



def getEncoder_oneBranch(partPoints_input, dis2Joint, Sigma,  is_training, reuse,  scope_predix="part_1", verbose=True,bn_decay=None):

    scname = scope_predix+"_pointNetPP_encoder"
    with tf.variable_scope( scname  ) as sc:
        if reuse:
            sc.reuse_variables()

        is_training = tf.constant(is_training, dtype=tf.bool)


        l0_xyz = partPoints_input
        l0_points = None


        print( "partPoints_input.shape = ", partPoints_input.shape )
        if partPoints_input.shape[2] == 6:
            l0_xyz    = partPoints_input[:,:, 0:3]
            l0_points = partPoints_input[:,:, 3:6]

    
        # assume self.shapeBatchSize==1
        inputDims = tf.reduce_max(l0_xyz , axis=[0, 1] ) - tf.reduce_min( l0_xyz , axis=[0, 1] )
        print(  "inputDims.shape = ",inputDims.shape  )
        does_part_exist =  tf.dtypes.cast( tf.reduce_mean( tf.abs(inputDims) ) > 0.01,  tf.float32 )

        
        # Set Abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=256, radius=0.1, nsample=128,
                                                            mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                            is_training=is_training, bn_decay=bn_decay, scope='layer1', bn=False)

        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.2, nsample=128,
                                                            mlp=[128, 128, 128], mlp2=None, group_all=False,
                                                            is_training=is_training, bn_decay=bn_decay, scope='layer2', bn=False)

        l4_xyz, l4_points, l4_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None,
                                                            mlp=[128, 128, 128], mlp2=None, group_all=True,
                                                            is_training=is_training, bn_decay=bn_decay, scope='layerxxx', bn=False)


        output_4 = tf.reshape(l4_points, [l0_xyz.shape[0], 128] )


        print(('output_4.shape = %s', output_4.shape))

        return output_4 * does_part_exist



def get_pointNet_code_of_singlePart(partPoints_input, dis2Joint, is_training, reuse,  scope_predix="part_1", verbose=True, bn_decay=None):
    
    
    if len(partPoints_input.shape)==2:
        partPoints_input = tf.expand_dims(partPoints_input, 0)
    
    partPoints_input_local = partPoints_input[:, :512, :]

    latent_code_1 = getEncoder_oneBranch_local( partPoints_input_local , None, 512,   is_training, reuse,  scope_predix+"_B1", verbose, bn_decay)
    latent_code_2 = getEncoder_oneBranch(       partPoints_input ,       None, 2048,   is_training, reuse,  scope_predix+"_B2", verbose, bn_decay)

    latent_code_12 = tf.concat([latent_code_1, latent_code_2], axis=1 )

    return latent_code_12

