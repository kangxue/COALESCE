
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import h5py
import cv2
import mcubes
import datetime
import shutil

from ops import *
from pointnet_plusplus_encoder_1B import get_pointNet_code_of_singlePart
from func_utils import *

import random
import time
import inspect

def lineno():
    """Returns the current line number in our program."""
    return str( inspect.currentframe().f_back.f_lineno )

class partAlignModel(object):
	def __init__(self, sess, pointSampleDim,  is_training = False,  \
					dataset_name='default', checkpoint_dir=None, data_dir='./data',    num_of_parts=4, \
					reWei=0, shapeBatchSize=16, FLAGS=None ):



		self.workdir =  os.path.split( data_dir )[0]
		if self.workdir[:2] == "./":
			self.workdir = self.workdir[2:]

		print("self.workdir = ", self.workdir)

		part_name_list, part_mesh_dir = getMeshInfo( self.workdir,  data_dir)
		self.part_name_list  =  part_name_list
		self.part_mesh_dir  = part_mesh_dir

		print( "self.part_name_list = ",  self.part_name_list )
		print( "self.part_mesh_dir = ",  self.part_mesh_dir )


		self.sess = sess

		self.ptsSampDim = pointSampleDim			
		
		self.input_dim = 128

		self.dataset_name = dataset_name
		self.checkpoint_dir = self.workdir + '/' + checkpoint_dir + "_partAlign"
		self.res_sample_dir = self.checkpoint_dir + "/res_sample/"
		self.test_res_dir = self.workdir + '/' + "test_res_partAlign"
		self.data_dir = data_dir
		

		self.num_of_parts = num_of_parts
		self.reWei = reWei
		self.shapeBatchSize = shapeBatchSize

		if not is_training:
			self.shapeBatchSize = 1
			shapeBatchSize = 1
		
		if not os.path.exists(  self.workdir   ):
			os.makedirs(  self.workdir  )

		if not os.path.exists(  self.checkpoint_dir   ):
			os.makedirs(  self.checkpoint_dir  )

		if not os.path.exists(  self.res_sample_dir   ):
			os.makedirs(  self.res_sample_dir  )

		if not os.path.exists(  self.test_res_dir   ):
			os.makedirs(  self.test_res_dir  )

			
		if os.path.exists(self.data_dir+  "/"+self.dataset_name+'.hdf5') :   
			
			self.data_dict = h5py.File(self.data_dir+  "/" + self.dataset_name+'.hdf5', 'r')

			points_shape = self.data_dict['allpartPts2048_ori'].shape

			self.trainSize = int( points_shape[0] * 0.8 )	
			trainSize = self.trainSize 	
			
			self.testSize = points_shape[0] - trainSize
			

			##### note:   we use point cloud as input
			self.ptsBatchSize = 2048*self.num_of_parts
			self.batch_size = 2048*self.num_of_parts

							
			print( "self.num_of_parts = ", self.num_of_parts )
			print( "self.ptsBatchSize = ", self.ptsBatchSize )
			print( "self.trainSize = ", self.trainSize )
			print( "self.trainSize = ", self.trainSize )

			# load obj list
			self.objlist_path = self.data_dir+'/'+self.dataset_name+'_obj_list.txt'
			if os.path.exists( self.objlist_path  ):
				text_file = open(self.objlist_path, "r")
				self.objlist = text_file.readlines()
					
				for i in range(len(self.objlist)):
					self.objlist[i] = self.objlist[i].rstrip('\r\n')
						
				text_file.close()
			else:
				print("error: cannot load "+ self.objlist_path)
				exit(0)
			

			
			######################## extract datasets
			if is_training:


				self.allpartPts2048_ori   = self.data_dict['allpartPts2048_ori_sorted'][:trainSize]
				self.allpartPts2048_ori_normal   = self.data_dict['allpartPts2048_ori_normal_sorted'][:trainSize]
				self.allpartPts2048_ori_dis   = self.data_dict['allpartPts2048_ori_dis_sorted'][:trainSize]

				self.objlist = self.objlist[:trainSize]
				self.data_examples_dir =  self.checkpoint_dir + "/train_examples_10/"

			else:
				
				self.allpartPts2048_ori   = self.data_dict['allpartPts2048_ori_sorted'][trainSize:]
				self.allpartPts2048_ori_normal   = self.data_dict['allpartPts2048_ori_normal_sorted'][trainSize:]
				self.allpartPts2048_ori_dis   = self.data_dict['allpartPts2048_ori_dis_sorted'][trainSize:]

				self.objlist = self.objlist[trainSize:]
				self.data_examples_dir = self.checkpoint_dir + "/test_examples_10/"

				### load point clouds before erosion
				self.data_dict_noEr = h5py.File(self.data_dir.split("_erode")[0] + "_erode0.0_256_ptsSorted/" + self.dataset_name+'.hdf5', 'r')
				self.allpartPts2048_ori_noEr          = self.data_dict_noEr['allpartPts2048_ori_sorted'][trainSize:]
				self.allpartPts2048_ori_normal_noEr   = self.data_dict_noEr['allpartPts2048_ori_normal_sorted'][trainSize:]
				self.data_dict_noEr.close()




			print(  "self.allpartPts2048_ori.shape = ", self.allpartPts2048_ori.shape  )
			outputExamples = 1
			if outputExamples:
				
				if not os.path.exists(  self.data_examples_dir   ):
					os.makedirs(  self.data_examples_dir  )
					
				for t in range(10):
					print(self.objlist[t])

					lables = np.ones(  2048*self.num_of_parts, dtype=np.uint8 )
					for lb in range(1, self.num_of_parts+1):
						lables[(lb-1)*2048 : lb*2048 ] = lb
					self.allpartPts2048_labels = lables		


					write_colored_ply( self.data_examples_dir + str(t) + '-' + self.objlist[t]+"_ori.ply",     self.allpartPts2048_ori[t],  self.allpartPts2048_ori_normal[t],   self.allpartPts2048_labels )

				
			self.data_dict.close()
			self.data_dict = None

		else:
			if is_training:
				print("error: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
				print("error: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
				print("error: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
				exit(0)
			else:
				print("warning: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
				print("warning: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
				print("warning: cannot load "+self.data_dir+'/'+self.dataset_name+'.hdf5')
		
		
		
		self.build_model()		



	def build_model(self):
		
		############# placeholders
		
		self.point_in_nml  = tf.placeholder( shape=[self.shapeBatchSize, self.num_of_parts*2048,   3], dtype=tf.float32 )
		self.normal_in     = tf.placeholder( shape=[self.shapeBatchSize,  self.num_of_parts*2048,   3], dtype=tf.float32 )

		self.point_gt = tf.placeholder(shape=[self.shapeBatchSize,  self.num_of_parts*2048,   3], dtype=tf.float32) 
		self.dis2joint = tf.placeholder(shape=[self.shapeBatchSize,  self.num_of_parts*2048    ],  dtype=tf.float32)  

		# for test only
		self.partMesh_vert =  tf.placeholder(shape=[self.shapeBatchSize, self.num_of_parts, 1000000,3], dtype=tf.float32) 
		##############
		
		
		self.inputPC_pt_norm = tf.concat(  [self.point_in_nml, self.normal_in] ,  axis=2 )


		self.part_code_list   = []
		self.part_code_list_s = []		
		for pid in range(self.num_of_parts):
			scope_predix="part_"+str(pid+1)
			self.part_code_list.append(    get_pointNet_code_of_singlePart( self.inputPC_pt_norm[:, pid*2048:(pid+1)*2048,: ],     self.allpartPts2048_ori_dis[:, pid*2048:(pid+1)*2048 ],      scope_predix=scope_predix,  is_training=True, reuse=False)  )
			self.part_code_list_s.append(  get_pointNet_code_of_singlePart( self.inputPC_pt_norm[:,  pid*2048:(pid+1)*2048,: ],    self.allpartPts2048_ori_dis[:, pid*2048:(pid+1)*2048 ],    scope_predix=scope_predix,  is_training=False, reuse=True)  )
		

		self.globalCode_1  = tf.concat(self.part_code_list,   axis=1 )  
		self.globalCode_1s = tf.concat(self.part_code_list_s, axis=1 )  
		print( "globalCode_1.shape = ", self.globalCode_1.shape )


		scope_predix="decoder_part"
		self.T1,  self.S1        =   self.generator(  self.globalCode_1,      scope_predix=scope_predix,  phase_train=True, reuse=False)  
		self.T1s, self.S1s       =   self.generator(  self.globalCode_1s,     scope_predix=scope_predix,  phase_train=False, reuse=True)  
		
		trans_1   = tf.reshape( tf.tile( tf.expand_dims( self.T1, 2) ,   [1,1,2048,1] ),  self.point_in_nml.shape )
		scales_1  = tf.reshape( tf.tile( tf.expand_dims( self.S1, 2) ,   [1,1,2048,3] ),  self.point_in_nml.shape )

		trans_1s  = tf.reshape( tf.tile( tf.expand_dims( self.T1s, 2) ,  [1,1,2048,1] ),  self.point_in_nml.shape )
		scales_1s = tf.reshape( tf.tile( tf.expand_dims( self.S1s, 2) ,  [1,1,2048,3] ),  self.point_in_nml.shape )

		trans_verts = tf.reshape(  tf.tile( tf.expand_dims( self.T1s, 2) ,  [1,1,1000000,1] ),  self.partMesh_vert.shape )
		scale_verts = tf.reshape(  tf.tile( tf.expand_dims( self.S1s, 2) ,  [1,1,1000000,3] ),  self.partMesh_vert.shape )


		self.O1  =  tf.multiply( self.point_in_nml, scales_1 ) + trans_1
		self.O1s =  tf.multiply( self.point_in_nml, scales_1s ) + trans_1s
		self.OVerts =  tf.multiply( self.partMesh_vert, scale_verts ) + trans_verts  
		

		self.ptsWeight = tf.ones(self.dis2joint.shape,  dtype=tf.float32  )
		if self.reWei:
			self.ptsWeight = tf.exp( tf.square( (self.dis2joint / 0.05) ) * -1 )

		self.ptsWeight = tf.tile(  tf.expand_dims( self.ptsWeight, 2 ),  [1,1,3] )


		self.loss_1  =tf.reduce_sum( tf.multiply(    tf.square(   self.O1  - self.point_gt ) , self.ptsWeight ) )
		self.loss_1s =tf.reduce_sum( tf.multiply(    tf.square(  self.O1s  - self.point_gt ) , self.ptsWeight ) )

		self.error_eva =tf.reduce_mean (   tf.sqrt( tf.reduce_sum(   tf.square( self.O1s - self.point_gt ), 	axis=2  ) ) )

		
		self.trainingVariables_encoders = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'part_'  )
		self.trainingVariables_decoders = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'decoder_part'  )


		print("################ trainingVariables_encoders: " )
		for v in self.trainingVariables_encoders:
			print(v.name)

		print("################ trainingVariables_decoders: " )
		for v in self.trainingVariables_decoders:
			print(v.name)

		self.saver = tf.train.Saver(max_to_keep=20, var_list=self.trainingVariables_encoders + self.trainingVariables_decoders )
		
		
	def generator(self,  z,   scope_predix="part_1", phase_train=True, reuse=False):
		with tf.variable_scope(scope_predix+"_simple_net") as scope:
			if reuse:
				scope.reuse_variables()

			print( "z.shape = ", z.shape )

			self.gf_dim = 32 
			
			h1 = lrelu(linear(z, self.gf_dim*16, 'h1_lin'))
			h1 = tf.concat([h1,z],1)
			
			h2 = lrelu(linear(h1, self.gf_dim*8, 'h4_lin'))
			h2 = tf.concat([h2,z],1)
			
			h3 = lrelu(linear(h2, self.gf_dim*4, 'h5_lin'))
			h3 = tf.concat([h3,z],1)
			
			h4 = lrelu(linear(h3, self.gf_dim*2, 'h6_lin'))
			h4 = tf.concat([h4,z],1)
			
			h6 = linear(h4,  4*self.num_of_parts, 'h8_lin')

			TS = tf.reshape(h6, [-1, self.num_of_parts, 4])


			return TS[:, :, :3], TS[:, :, 3:4]



	def train(self,  learning_rate, beta1, for_debug, epochNum ):


		ae_optim_iter1 = tf.train.AdamOptimizer( learning_rate, beta1=beta1).minimize(self.loss_1,         var_list=self.trainingVariables_encoders+self.trainingVariables_decoders )
		

		self.sess.run(tf.global_variables_initializer())
		

		batch_index_list = np.arange( self.allpartPts2048_ori.shape[0] )

		print("len(batch_index_list) = ", len(batch_index_list) )

		num_train = len(batch_index_list) 
		num_train = num_train - num_train % self.shapeBatchSize


		if self.ptsBatchSize%self.batch_size != 0:
			print("ptsBatchSize % batch_size != 0")
			exit(0)
		
		counter = 0
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter+1
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")


		print(  "num_train = ", num_train)
		
		print(  "self.allpartPts2048_ori.shape[0]  = ", self.allpartPts2048_ori.shape[0] )
		print(  "self.allpartPts2048_ori.shape  = ", self.allpartPts2048_ori.shape )
		


		random.seed(  int(time.time()) )
		for epoch in range(counter, epochNum):


			ae_optim = ae_optim_iter1

			np.random.shuffle(batch_index_list)
			avg_loss1  = 0
			avg_loss2  = 0
			avg_loss3  = 0

			avg_num = 0
			ignoreCount = 0
			
			for idx in range(0, num_train,  self.shapeBatchSize):

				dxb = batch_index_list[ idx:idx+self.shapeBatchSize ]

				inputPC_list = []
				inputPC_normal_list = []
				for lb in range( 1, self.num_of_parts+1  ):
					inputPC_list.append(   normalizePointCloud( self.allpartPts2048_ori[ dxb,  ( lb-1)*2048 : lb*2048 ] )   )
					inputPC_normal_list.append(   self.allpartPts2048_ori_normal[ dxb,  ( lb-1)*2048 : lb*2048 ]   )
							
				############ update
				_, err1 = self.sess.run([ae_optim, self.loss_1  ],  
				feed_dict={
					self.point_in_nml: np.concatenate( inputPC_list, axis=1 ),
					self.normal_in: np.concatenate( inputPC_normal_list, axis=1 ),  
					self.point_gt:  self.allpartPts2048_ori[ dxb],
					self.dis2joint:   self.allpartPts2048_ori_dis[ dxb]
				})

				avg_loss1  += err1
				avg_num += 1
				if (idx%16 == 0):
					print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
					print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f,   avgloss: %.8f " % \
							(epoch, epochNum, idx, num_train, time.time() - start_time,   avg_loss1/avg_num ) )
					

				##########################################################
				
				if idx==(num_train-self.shapeBatchSize):
					if epoch%5 == 4 :
						self.save(self.checkpoint_dir, epoch)

					dxb = batch_index_list[ idx:idx+self.shapeBatchSize ]

					inputPC_list = []
					inputPC_normal_list = []
					for lb in range( 1, self.num_of_parts+1  ):
						inputPC_list.append(   normalizePointCloud( self.allpartPts2048_ori[ dxb,  ( lb-1)*2048 : lb*2048 ] )   )
						inputPC_normal_list.append(   self.allpartPts2048_ori_normal[ dxb,  ( lb-1)*2048 : lb*2048 ]   )
											
					# Update AE network
					out1  = self.sess.run( self.O1s,
						feed_dict={
							self.point_in_nml: np.concatenate( inputPC_list, axis=1 ),
							self.normal_in: np.concatenate( inputPC_normal_list, axis=1 ),  
							self.point_gt: self.allpartPts2048_ori[ dxb],
							self.dis2joint:   self.allpartPts2048_ori_dis[ dxb]
						})						


					input_pts = np.concatenate( inputPC_list, axis=1 )
					gt_normals = np.concatenate( inputPC_normal_list, axis=1 )
	
					write_colored_ply( self.res_sample_dir+  str(epoch)  +".input_parts.ply",   input_pts[0],   gt_normals[0],   self.allpartPts2048_labels )
					write_colored_ply( self.res_sample_dir+  str(epoch)  +".output_parts.ply",  out1[0],        gt_normals[0],   self.allpartPts2048_labels )
					write_colored_ply( self.res_sample_dir+  str(epoch)  +".gt_parts.ply",  self.allpartPts2048_ori[ dxb[0] ],        gt_normals[0],   self.allpartPts2048_labels )



	def transform_vertices(self,  inputPC_list, inputPC_normal_list, part_meshVert_list  ):

		## pad vertices with 0 and test
		PartVert_padded_List = []
		for pid in range( self.num_of_parts ):
			PartVert_padded = np.zeros( [1, 1000000,3],  np.float32 )
			PartVert_padded[0, : part_meshVert_list[pid].shape[0], :] = part_meshVert_list[pid]
			PartVert_padded_List.append( PartVert_padded )
	
		
		outPartVert_padded  = self.sess.run(  self.OVerts,  
			feed_dict={
				self.point_in_nml:   np.expand_dims(  np.vstack( inputPC_list ),        axis=0 ),
				self.normal_in:      np.expand_dims(  np.vstack( inputPC_normal_list ), axis=0 ),
				self.partMesh_vert:  np.expand_dims(  np.vstack( PartVert_padded_List ), axis=0 )
			})
				
		Out_PartVert_List = []
		for pid in range( self.num_of_parts ):
			Out_PartVert_List.append(   outPartVert_padded[0, pid,  :part_meshVert_list[pid].shape[0], :]     )

		return Out_PartVert_List



	def test(self, specEpoch, differentShapes=1 ):


		could_load, checkpoint_counter = self.load_fortest(self.checkpoint_dir, specEpoch)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		self.test_res_dir =  self.test_res_dir  + '_Epoch' + str(checkpoint_counter+1) + '/'


		if differentShapes:
			self.test_res_dir =  self.test_res_dir[:-1] + '_diffshapes/'


		if not os.path.exists(  self.test_res_dir   ):
			os.makedirs(  self.test_res_dir  )

		rdIdPath = self.workdir + "/" + self.workdir + "_rdid.txt"

		if  not os.path.exists(  rdIdPath  ):
			print( rdIdPath  + "doesn't exist!")
			exit(0)

		rdIdx_array = ( np.loadtxt( rdIdPath  ) ).astype(np.int32)
		print("rdIdx_array.shape = ", rdIdx_array.shape )
		print(rdIdx_array)

		random.seed(  0 )

		testNum = self.testSize

		if differentShapes:
			testNum = min( rdIdx_array.shape[0], 3000 )  

		print( "testNum = ",  testNum )

		
		for t in range(  testNum  ):
			

			print("t = " , t)


			if not differentShapes:
				allpartPts = np.copy( self.allpartPts2048_ori[t] )
				allpartPts_normal = np.copy( self.allpartPts2048_ori_normal[t] )
			else:
				allpartPts = np.copy( self.allpartPts2048_ori[0] )
				allpartPts_normal = np.copy( self.allpartPts2048_ori_normal[0] )
				

			part_meshVert_list = []
			part_meshTria_list = []				
			part_oriModelName_list = []		
			meshPath_list = []	

			part_mesh0000Vert_list = []
			part_mesh0000Tria_list = []
			mesh0000Path_list = []	

			part_PC_NoEr_list  = []
			part_PCNormal_NoEr_list  = []

			if differentShapes:
				for pid in range( self.num_of_parts ):

					rdId = rdIdx_array[t, pid]  #random.randrange(100)
					allpartPts[ pid*2048:(pid+1)*2048,:] = self.allpartPts2048_ori[rdId, pid*2048:(pid+1)*2048,:]  
					allpartPts_normal[ pid*2048:(pid+1)*2048,:] = self.allpartPts2048_ori_normal[rdId, pid*2048:(pid+1)*2048,:]  
							

					print("rdId = ", rdId)
					mesh_path = self.part_mesh_dir + self.objlist[rdId] + '/' + self.part_name_list[pid] + '.ply'
					print(mesh_path)
			
					vertices, tri_data = load_ply_withfaces( mesh_path )
						
					part_meshVert_list.append(vertices)
					part_meshTria_list.append(tri_data)
					part_oriModelName_list.append(self.objlist[rdId])
					meshPath_list.append(  mesh_path )

					mesh0000_path = self.part_mesh_dir[:-14] + 'part_mesh0000/' + self.objlist[rdId] + '/' + self.part_name_list[pid] + '.ply'
					vertices, tri_data = load_ply_withfaces( mesh0000_path )
					part_mesh0000Vert_list.append(vertices)
					part_mesh0000Tria_list.append(tri_data)
					mesh0000Path_list.append( mesh0000_path )


					part_PC_NoEr_list.append(        self.allpartPts2048_ori_noEr[       rdId, pid*2048:(pid+1)*2048,:] )
					part_PCNormal_NoEr_list.append(  self.allpartPts2048_ori_normal_noEr[rdId, pid*2048:(pid+1)*2048,:] )


			else:

				for pid in range( self.num_of_parts ):

					mesh_path = self.part_mesh_dir + self.objlist[t] + '/' + self.part_name_list[pid] + '.ply'
					print(mesh_path)
								
					vertices, tri_data = load_ply_withfaces( mesh_path )
					
					part_meshVert_list.append(vertices)
					part_meshTria_list.append(tri_data)
					part_oriModelName_list.append(self.objlist[t])
					meshPath_list.append(  mesh_path )

					
					mesh0000_path = self.part_mesh_dir[:-14] + 'part_mesh0000/'+ self.objlist[t] + '/' + self.part_name_list[pid] + '.ply'
					vertices, tri_data = load_ply_withfaces( mesh0000_path )
					part_mesh0000Vert_list.append(vertices)
					part_mesh0000Tria_list.append(tri_data)
					mesh0000Path_list.append( mesh0000_path )

					part_PC_NoEr_list.append(        self.allpartPts2048_ori_noEr[       t,  pid*2048:(pid+1)*2048,:] )
					part_PCNormal_NoEr_list.append(  self.allpartPts2048_ori_normal_noEr[t,  pid*2048:(pid+1)*2048,:]  )

			
			#if "cochair4part" in self.dataset_name:
			#	for pid in range( self.num_of_parts ):
			#		part_meshVert_list[pid] =  np.multiply(  part_meshVert_list[pid],    np.array([1,1, -1], np.float32 ) )   



			####### get test input
			inputPC_list = []
			GT_list = []
			inputPC_normal_list = []
						
			for lb in range( 1, self.num_of_parts+1  ):            

				inputPC  = allpartPts[ ( lb-1)*2048 : lb*2048 ] 
				if doesPartExist(  inputPC  ):
					_, part_mesh0000Vert_list[lb-1] = normalizePointCloud_withMeshVert(  inputPC,   part_mesh0000Vert_list[lb-1]  ) 
					_, part_PC_NoEr_list[lb-1] = normalizePointCloud_withMeshVert(  inputPC,   part_PC_NoEr_list[lb-1]  ) 
					inputPC, part_meshVert_list[lb-1] = normalizePointCloud_withMeshVert(  inputPC,   part_meshVert_list[lb-1]  ) 

				inputPC_list.append( inputPC )
				GT_list.append(   allpartPts[ ( lb-1)*2048 : lb*2048 ]  )
                
				inputPC_normal_list.append(   allpartPts_normal[ ( lb-1)*2048 : lb*2048 ]   )

				
			for pid in range(self.num_of_parts):

				if  doesPartExist( inputPC_list[pid]  )  and   part_meshVert_list[pid].shape[0] == 0:
					print( "point exist but mesh not exist!!!!!!!!! bug")
					print("mesh path: ",  meshPath_list[ pid ]   )
					exit(0)

		
			##  test
		
			out1  = self.sess.run( self.O1s ,  
				feed_dict={
					self.point_in_nml:   np.expand_dims(  np.vstack( inputPC_list ),        axis=0 ),
					self.normal_in:      np.expand_dims(  np.vstack( inputPC_normal_list ), axis=0 )
				})
					

			Out_PartVert_List = self.transform_vertices( inputPC_list, inputPC_normal_list,  part_meshVert_list )

			Out_PartVert0000_List = self.transform_vertices( inputPC_list, inputPC_normal_list,  part_mesh0000Vert_list )

			out_PC_NoEr_list = self.transform_vertices( inputPC_list, inputPC_normal_list,  part_PC_NoEr_list )


			###############################
			write_colored_ply( self.test_res_dir+  str(t)  +"-input.ply",       np.vstack( inputPC_list ),  np.vstack( inputPC_normal_list ),  self.allpartPts2048_labels )
			write_colored_ply( self.test_res_dir+  str(t)  +"-output.ply",      out1[0],                    np.vstack( inputPC_normal_list ),  self.allpartPts2048_labels )
			write_colored_ply( self.test_res_dir+  str(t)  +"-output_noEr.ply", np.vstack( out_PC_NoEr_list ),     np.vstack( part_PCNormal_NoEr_list ),  self.allpartPts2048_labels )
			
			if not differentShapes:
				write_colored_ply( self.test_res_dir+  str(t)  +".output_gt.ply",       allpartPts,                 allpartPts_normal,  self.allpartPts2048_labels )
				

			# output mesh data
			
			part_mesh_outdir = self.test_res_dir+  str(t)  + '/'
			if  os.path.exists( part_mesh_outdir  ):
				shutil.rmtree( part_mesh_outdir )
			os.makedirs( part_mesh_outdir  )

			for pid in range(self.num_of_parts):
				if part_meshVert_list[pid].shape[0] > 0 :
					write_ply_withfaces( part_mesh_outdir + self.part_name_list[pid] + ".ply",   \
																Out_PartVert_List[pid],  part_meshTria_list[pid]  )

			with open(  part_mesh_outdir +   str(t)  +".name.txt", 'w' ) as file:
				for name in part_oriModelName_list:
				    file.write(  name + '\n' )
                
			# output input mesh
			part_mesh_outdir = self.test_res_dir+  str(t)  + '_input/'
			if  os.path.exists( part_mesh_outdir  ):
				shutil.rmtree( part_mesh_outdir )
			os.makedirs( part_mesh_outdir  )

			for pid in range(self.num_of_parts):
				if part_meshVert_list[pid].shape[0] > 0 :
					write_ply_withfaces( part_mesh_outdir + self.part_name_list[pid] + ".ply",   \
																part_meshVert_list[pid],  part_meshTria_list[pid]  )
																
			mergedVert, mergedFaces = mergeMesh_parts( part_meshVert_list, part_meshTria_list  )
			write_ply_withfaces( part_mesh_outdir +  str(t)  +"-input_mesh.ply",   mergedVert, mergedFaces  )


			########## output merged mesh data
			mergedVert, mergedFaces = mergeMesh_parts( Out_PartVert_List, part_meshTria_list  )
			write_ply_withfaces( self.test_res_dir+  str(t)  +"-output_mesh.ply",  mergedVert,  mergedFaces  )
	

			mergedVert0000, mergedFaces0000 = mergeMesh_parts( Out_PartVert0000_List, part_mesh0000Tria_list  )
			write_ply_withfaces( self.test_res_dir+  str(t)  +"-output_mesh0000.ply",  mergedVert0000,  mergedFaces0000  )
	

	@property
	def model_dir(self):
		return "{}_{}".format(
				self.dataset_name, self.input_dim)
			
	def save(self, checkpoint_dir, step):
		model_name = "model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

	def load_fortest(self, checkpoint_dir, specEpoch):
		import re
		print(" [*] Reading checkpoints...")
		checkpoint_path = os.path.join(checkpoint_dir, self.model_dir, 'model-'+str(specEpoch-1) )
		
		if os.path.exists( checkpoint_path+'.index' ):
			self.saver.restore(self.sess,  checkpoint_path)
			print(" [*] Success to read {}".format( os.path.basename(checkpoint_path ))  )
			return True,specEpoch-1
		else:
			print(" [*] Failed to find a checkpoint: " + checkpoint_path)
			return False, 0
