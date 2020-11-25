
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

from ops import *
from pointnet_plusplus_encoder import get_pointNet_code_of_singlePart
from func_utils import *

import random
import time

from partAE_model import PAE
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from distutils.dir_util import copy_tree
	

class JSM(object):
	def __init__(self, sess, pointSampleDim,  is_training = False,  \
					dataset_name='default', checkpoint_dir=None, data_dir='./data', rdDisplaceRange=0.0,  rdScaleRange=0.1,  num_of_parts=4, FLAGS=None  ):


		self.PAE_list = []
		for partLabel in range(1, num_of_parts+1):

			print("============================================================")
			print("=========== intialize PAE  for part ", partLabel)
			pae = PAE(					
					sess,
					pointSampleDim,
					FLAGS,
					is_training = False,
					dataset_name=dataset_name,
					checkpoint_dir=checkpoint_dir,
					data_dir=data_dir,
					rdDisplaceRange=rdDisplaceRange,
					rdScaleRange=rdScaleRange,
					partLabel=partLabel,
					loadDataset=False,
					shapeBatchSize=1 )
					

			could_load, checkpoint_counter = pae.load(pae.checkpoint_dir)
			if could_load:
				print( "part ", partLabel, ", checpoint load success:  epoch " , checkpoint_counter)
			else:
				print( "part ", partLabel, ", checpoint load failed!!!!!!!" )

			self.PAE_list.append(  pae )


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
		
		self.input_dim = pointSampleDim 

		self.dataset_name = dataset_name
		self.checkpoint_dir = self.workdir + '/' + checkpoint_dir + "_jointSynthesis" 
		self.res_sample_dir = self.checkpoint_dir + "/res_sample/"
		self.test_res_dir = self.workdir + '/' + "test_res_jointSynthesis/"
		self.data_dir = data_dir
		self.rdDisplaceRange = rdDisplaceRange
		self.rdScaleRange = rdScaleRange



		self.num_of_parts = num_of_parts
		

		self.is_training = is_training

		if not os.path.exists(  self.workdir   ):
			os.makedirs(  self.workdir  )

		if not os.path.exists(  self.checkpoint_dir   ):
			os.makedirs(  self.checkpoint_dir  )

		if not os.path.exists(  self.res_sample_dir   ):
			os.makedirs(  self.res_sample_dir  )

		if not os.path.exists(  self.test_res_dir   ):
			os.makedirs(  self.test_res_dir  )

			
		if os.path.exists(self.data_dir+'/'+self.dataset_name+'.hdf5') :   
			self.data_dict = h5py.File(self.data_dir+  "/" + self.dataset_name+'.hdf5', 'r')

			self.ptsBatchSize = FLAGS.ptsBatchSize 
			points_shape = self.data_dict['points_'+str(self.ptsSampDim)+'_aroundjoint'][:, :self.ptsBatchSize, :].shape

			self.trainSize = int( points_shape[0] * 0.8 )	
			trainSize = self.trainSize 	
			self.testSize = points_shape[0] - trainSize

			self.batch_size = self.ptsBatchSize

				
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

				self.data_points_wholeShape = self.data_dict['points_'+str(self.ptsSampDim) + '_aroundjoint'][:trainSize, :self.ptsBatchSize, :]
				self.data_values_wholeShape = self.data_dict['jointValues_'+str(self.ptsSampDim) + '_aroundjoint'][:trainSize, :self.ptsBatchSize, :]

				self.allpartPts2048_ori   = self.data_dict['allpartPts2048_ori_sorted'][:trainSize]
				self.allpartPts2048_ori_normal   = self.data_dict['allpartPts2048_ori_normal_sorted'][:trainSize]
				self.allpartPts2048_ori_dis      = self.data_dict['allpartPts2048_ori_dis_sorted'][:trainSize]

				
				self.objlist = self.objlist[:trainSize]
				self.data_examples_dir =  self.checkpoint_dir + "/train_examples_10/"

			else:
								
				self.data_points_wholeShape = self.data_dict['points_'+str(self.ptsSampDim) + '_aroundjoint'][trainSize:, :self.ptsBatchSize, :]
				self.data_values_wholeShape = self.data_dict['jointValues_'+str(self.ptsSampDim) + '_aroundjoint'][trainSize:, :self.ptsBatchSize, :]
				
				self.allpartPts2048_ori   = self.data_dict['allpartPts2048_ori_sorted'][trainSize:]
				self.allpartPts2048_ori_normal   = self.data_dict['allpartPts2048_ori_normal_sorted'][trainSize:]
				self.allpartPts2048_ori_dis      = self.data_dict['allpartPts2048_ori_dis_sorted'][trainSize:]
									

				self.objlist = self.objlist[trainSize:]
				self.data_examples_dir = self.checkpoint_dir + "/test_examples_10/"

				self.data_voxels_128 = self.data_dict['voxels_128' ][trainSize:] 


			######################## ouput samples of traing or test data
			print(  "self.data_points_wholeShape.shape = ", self.data_points_wholeShape.shape  )
			print(  "self.data_values_wholeShape.shape = ", self.data_values_wholeShape.shape  )
			print(  "self.allpartPts2048_ori.shape = ", self.allpartPts2048_ori.shape  )
			#print(  "self.singlePartPoints2048.shape = ", self.singlePartPoints2048.shape  )
			outputExamples = 1
			if outputExamples:
				
				if not os.path.exists(  self.data_examples_dir   ):
					os.makedirs(  self.data_examples_dir  )
					
				for t in range(5):
					print(self.objlist[t])

					lables = np.ones(  self.num_of_parts*2048, dtype=np.uint8 )
					for lb in range(1, self.num_of_parts+1):
						lables[(lb-1)*2048 : lb*2048 ] = lb
					self.allpartPts2048_labels = lables		

					write_colored_ply( self.data_examples_dir + self.objlist[t]+"_samples.ply", \
										intGridNodes_2_floatPoints( self.data_points_wholeShape[t], self.ptsSampDim ), None, (np.squeeze(self.data_values_wholeShape[t])>0.5).astype(np.uint8) )
					write_colored_ply( self.data_examples_dir + self.objlist[t]+"_ori.ply",     self.allpartPts2048_ori[t],  self.allpartPts2048_ori_normal[t],  self.allpartPts2048_labels )
					write_colored_ply( self.data_examples_dir + self.objlist[t]+"_ori_disvis.ply",     self.allpartPts2048_ori[t],  self.allpartPts2048_ori_normal[t],  \
										( np.squeeze(self.allpartPts2048_ori_dis[t]) < 0.1).astype(np.uint8)  )
		
			########################
			if self.ptsBatchSize!=self.data_points_wholeShape.shape[1]:
				print("error: ptsBatchSize!=data_points_wholeShape.shape")
				exit(0)

				
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
		

		if not is_training:
			self.test_size = 32

			if self.test_size > self.ptsSampDim:
				self.test_size = self.ptsSampDim

			self.batch_size = self.test_size*self.test_size*self.test_size #do not change
			
		
		self.build_model()

		

	def build_model(self):
		

		self.point_in  = tf.placeholder(shape=[self.num_of_parts*2048, 3], dtype=tf.float32)  ##### placeholder, feed it with  point cloud
		self.normal_in = tf.placeholder(shape=[self.num_of_parts*2048, 3], dtype=tf.float32)  ##### placeholder, feed it with  point cloud normals
		self.distance_in = tf.placeholder(shape=[self.num_of_parts*2048  ], dtype=tf.float32)
		self.point_exist_mask = tf.placeholder(shape=[self.num_of_parts*2048  ], dtype=tf.float32)  # only use during test


		if self.is_training:

			self.point_in_after_transform = self.point_in 

		else:
			### for test time optimization

			self.part_translates = tf.get_variable("align_translation", [self.num_of_parts, 2 ])
			self.part_scales = tf.get_variable("align_scaling",     [self.num_of_parts, 3 ])

			self.part_translates_padded = tf.concat(  [ tf.constant(0.0, tf.float32, shape=[self.num_of_parts, 1] ),  self.part_translates ],   1   )
			translations = tf.tile( tf.expand_dims(self.part_translates_padded , 1 ),   [1, 2048, 1] )
			translations = tf.reshape(translations, [self.num_of_parts*2048, 3] )

			scales = tf.tile( tf.expand_dims(  self.part_scales, 1 ),   [1, 2048, 1] )
			scales = tf.reshape(scales, [self.num_of_parts*2048, 3] )

			print(" translations.shape = ",  translations.shape )
			print(" scales.shape = ",  scales.shape )


			self.point_in_after_transform = tf.multiply( self.point_in,  scales ) + translations


		self.inputPC_pt_norm = tf.concat(  [self.point_in_after_transform, self.normal_in] ,  axis=1 )

		self.distance_in_mod  = self.distance_in  +   tf.dtypes.cast(self.distance_in<0.0001   , tf.float32 ) * 1000

		topvalues,  topIdx =  tf.math.top_k(  -tf.squeeze(self.distance_in_mod),  k=1024 )
		self.pointsAroundJoints    = tf.gather(self.point_in_after_transform, topIdx)
		self.pointsAroundJoints_nml = tf.gather(self.normal_in, topIdx)


		####
		print( "self.distance_in.shape = " , self.distance_in.shape  ) 
		print( "topIdx.shape = " , topIdx.shape  ) 
		print( "self.pointsAroundJoints.shape = " , self.pointsAroundJoints.shape  ) 
		print( "self.pointsAroundJoints_nml.shape = " , self.pointsAroundJoints_nml.shape  ) 


		self.part_code_list   = []
		self.part_code_list_s = []		
		for pid in range(self.num_of_parts):
			scope_predix="part_"+str(pid+1)
			
			self.part_code_list.append(    get_pointNet_code_of_singlePart( self.inputPC_pt_norm[ pid*2048:(pid+1)*2048,: ],   self.distance_in[ pid*2048:(pid+1)*2048  ], 
																			scope_predix=scope_predix,  is_training=True, reuse=True)  )
			self.part_code_list_s.append(  get_pointNet_code_of_singlePart( self.inputPC_pt_norm[ pid*2048:(pid+1)*2048,: ],   self.distance_in[ pid*2048:(pid+1)*2048 ],        
																			scope_predix=scope_predix,  is_training=False, reuse=True)  )
		
		self.globalCode_1  = tf.concat(self.part_code_list,   axis=1 )  
		self.globalCode_1s = tf.concat(self.part_code_list_s, axis=1 )  
		print( "globalCode_1.shape = ", self.globalCode_1.shape )


		######## Generator
		self.point_coord = tf.placeholder(shape=[self.batch_size,3], dtype=tf.float32)  ##### placeholder
		self.point_value = tf.placeholder(shape=[self.batch_size,1], dtype=tf.float32)  ##### placeholder

		self.scope_predix = "wholeshape_"   ### this name is confusing. Although we decode implicit function for whole shape. However, due to the special point sampling schema, only joint area will be reconstructed

		self.G   = self.joint_generator(self.point_coord, self.point_value,    self.globalCode_1,      scope_predix=self.scope_predix,  phase_train=True, reuse=False)
		self.sG  = self.joint_generator(self.point_coord, self.point_value,    self.globalCode_1s,     scope_predix=self.scope_predix,  phase_train=False, reuse=True)


		self.mse_loss = tf.reduce_mean(tf.square(self.point_value - self.G))
		self.mse_loss_eva = tf.reduce_mean(  tf.square(self.point_value - self.sG)  ) 

        
		self.pointsAroundJoints_nml = tf.math.l2_normalize(  self.pointsAroundJoints_nml,  axis = 1 )
		self.points_0  = self.pointsAroundJoints + self.pointsAroundJoints_nml * 0.005
		self.points_1  = self.pointsAroundJoints - self.pointsAroundJoints_nml * 0.005
        
		self.G_0   = self.joint_generator(self.points_0, self.point_value,    self.globalCode_1,      scope_predix=self.scope_predix,  phase_train=True, reuse=True)
		self.G_1   = self.joint_generator(self.points_1, self.point_value,    self.globalCode_1,      scope_predix=self.scope_predix,  phase_train=True, reuse=True)
		

		self.match_loss = ( tf.reduce_mean(tf.square(self.G_0))   +  tf.reduce_mean(tf.square(self.G_1 - 1.0))  ) * 0.5   * 0.2

		self.match_loss_beforeWeight = self.match_loss
        
		self.loss_0 = self.mse_loss
		self.loss_1 = self.mse_loss + self.match_loss  


		self.fineTune_loss =  ( tf.reduce_mean(tf.abs(self.G_0))   +  tf.reduce_mean(tf.abs(self.G_1 - 1.0))  ) * 0.5 



		self.trainingVariables_encoders = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'part_'  )
		self.trainingVariables_decoders = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'wholeshape_'  )
		self.trainingVariables_align = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= 'align_'  )
		self.trainingVariables_lastEncodingLayer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=r'part_._pointNetPP_encoder/layerxxx/'  )
	
		

		print("################ trainingVariables_encoders: " )
		for v in self.trainingVariables_encoders:
			print(v.name)


		print("################ trainingVariables_decoders: " )
		for v in self.trainingVariables_decoders:
			print(v.name)


		print("################ trainingVariables_align: " )
		for v in self.trainingVariables_align:
			print(v.name)



		print("################ trainingVariables_lastEncodingLayer: " )
		for v in self.trainingVariables_lastEncodingLayer:
			print(v.name)


		self.saver = tf.train.Saver(max_to_keep=2000, var_list=self.trainingVariables_encoders + self.trainingVariables_decoders )
		
		
		
	def joint_generator(self, points, pvalues,  z,   scope_predix="joint", phase_train=True, reuse=False):
		with tf.variable_scope(scope_predix+"_simple_net") as scope:
			if reuse:
				scope.reuse_variables()

			
			print( "z.shape = ", z.shape )

			point_num = points.shape[0]

			self.gf_dim = 128
			zs = tf.tile(z,   [point_num, 1]  )

			pointz = tf.concat([points,zs],1)
			
			print("points",points.shape)
			print("pointz",pointz.shape)	
			
			h1 = lrelu(linear(pointz, self.gf_dim*16, 'h1_lin'))
			h1 = tf.concat([h1,pointz],1)
			
			h2 = lrelu(linear(h1, self.gf_dim*8, 'h4_lin'))
			h2 = tf.concat([h2,pointz],1)
			
			h3 = lrelu(linear(h2, self.gf_dim*4, 'h5_lin'))
			h3 = tf.concat([h3,pointz],1)
			
			h4 = lrelu(linear(h3, self.gf_dim*2, 'h6_lin'))
			h4 = tf.concat([h4,pointz],1)
			
			h5 = lrelu(linear(h4, self.gf_dim, 'h7_lin'))
			h6 = tf.nn.sigmoid(linear(h5, 1, 'h8_lin'))


			print( "points.shape = ", points.shape )
			print( "pvalues.shape = ", pvalues.shape )
			print( "h6.shape = ", h6.shape )

			
			return tf.reshape(h6, [point_num,1])
	

	def train(self,  learning_rate, beta1, for_debug, epochNum ):
				
		ae_optim_iter1 = tf.train.AdamOptimizer( learning_rate, beta1=beta1).minimize(self.mse_loss ,   var_list=self.trainingVariables_decoders )
		ae_optim_iter2 = tf.train.AdamOptimizer( learning_rate, beta1=beta1).minimize(self.mse_loss + self.match_loss ,         var_list=self.trainingVariables_encoders+self.trainingVariables_decoders )


		self.sess.run(tf.global_variables_initializer())
		

		# reload PAE_list
		for pid in range( self.num_of_parts):
			print("=========== load checkpoint for part ", pid+1)
			could_load, checkpoint_counter = self.PAE_list[pid].load(self.PAE_list[pid].checkpoint_dir)
			if could_load:
				print( "part ",  pid+1, ", checpoint load success:  epoch " , checkpoint_counter)
			else:
				print( "part ",  pid+1, ", checpoint load failed!!!!!!!" )
				exit(0)

		
		batch_index_list = np.arange( self.data_points_wholeShape.shape[0] )

		######################## data augment for chair back bars. See supplementary for details  ########################
		if "Chair" in self.data_dir:
			barchairList = []

			barChairObjList = loadObjNameList("03001627_augment_list.txt")
			for oname in barChairObjList:
				if oname in self.objlist:
					barchairList.append(  self.objlist.index(oname) )
					print( "add bar chair: ",  self.objlist[barchairList[-1]]  )

			batch_index_list = np.concatenate( ( batch_index_list, barchairList, barchairList, barchairList, barchairList ) )
			
		################################################################################################################

		numShapes = batch_index_list.shape[0]			

		print("batch_index_list = ", batch_index_list)
		print("numShapes = ", numShapes)


		if for_debug:
			numShapes = 1
			batch_index_list = np.arange(numShapes)


		batch_num = int(self.ptsBatchSize/self.batch_size)
		print("batch_num = ", batch_num)
		
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

		random.seed(  int(time.time()) )
		for epoch in range(counter, epochNum):


			if epoch < 80:
				ae_optim = ae_optim_iter1
				print( "ae_optim = ae_optim_iter1")
			elif epoch < 10000:
				ae_optim = ae_optim_iter2
				print( "ae_optim = ae_optim_iter2")
			else:
				ae_optim = ae_optim_iter2
				print( "ae_optim = ae_optim_iter2")



			np.random.shuffle(batch_index_list)
			avg_mseLoss = 0
			avg_matchLoss = 0
			avg_num = 0
			for idx in range(0, numShapes):
				for minib in range(batch_num):
					dxb = batch_index_list[idx]
					
					batch_points_int = self.data_points_wholeShape[dxb,  minib*self.batch_size:(minib+1)*self.batch_size]
					batch_points = intGridNodes_2_floatPoints( batch_points_int, self.ptsSampDim )
					batch_values = self.data_values_wholeShape[dxb, minib*self.batch_size:(minib+1)*self.batch_size]
					
					# augment dataset
					batch_points, inputPC_all = randomDisplaceScale_TwoPointSet( batch_points,  self.allpartPts2048_ori[ dxb, :, :] , \
																				displaceRange=self.rdDisplaceRange, scaleRange=self.rdScaleRange ) 

					# Update AE network
					_, mseLoss, matchLoss = self.sess.run([ae_optim, self.mse_loss,  self.match_loss_beforeWeight ],
						feed_dict={
							self.point_in: inputPC_all,
							self.normal_in: self.allpartPts2048_ori_normal[ dxb, :, : ],
							self.distance_in:  self.allpartPts2048_ori_dis[ dxb, : ],
							self.point_coord: batch_points,
							self.point_value: batch_values
						})
					avg_mseLoss += mseLoss
					avg_matchLoss += matchLoss
					avg_num += 1
					if (idx%16 == 0):
						print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
						print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, mseLoss: %.8f,  matchLoss: %.8f,   avg_mseLoss: %.8f,   avg_matchLoss: %.8f" \
							% (epoch, epochNum, idx, numShapes, time.time() - start_time, mseLoss, matchLoss, avg_mseLoss/avg_num, avg_matchLoss/avg_num))

				if idx==numShapes-1:
					model_float = np.zeros([self.ptsSampDim,self.ptsSampDim,self.ptsSampDim],np.float32)
					input_model_float = np.zeros([self.ptsSampDim,self.ptsSampDim,self.ptsSampDim],np.float32)
					for minib in range(batch_num):
						dxb = batch_index_list[idx]

							
						batch_points_int = self.data_points_wholeShape[dxb,  minib*self.batch_size:(minib+1)*self.batch_size]
						batch_points = intGridNodes_2_floatPoints( batch_points_int, self.ptsSampDim )
						batch_values = self.data_values_wholeShape[dxb, minib*self.batch_size:(minib+1)*self.batch_size]
							
						# augment dataset
						batch_points, inputPC_all = randomDisplaceScale_TwoPointSet( batch_points,  self.allpartPts2048_ori[ dxb, :, :] , \
																				displaceRange=self.rdDisplaceRange, scaleRange=self.rdScaleRange ) 

						model_out = self.sess.run(self.sG,
							feed_dict={
								self.point_in: inputPC_all,
								self.normal_in: self.allpartPts2048_ori_normal[ dxb, :, : ],
								self.distance_in:  self.allpartPts2048_ori_dis[ dxb, : ],
								self.point_coord: batch_points,
								self.point_value: batch_values
							})
							

						batch_points_int = floatPoints_2_uint8GridNodes( batch_points, self.ptsSampDim )

						print(  "self.ptsSampDim = ", self.ptsSampDim )
						print(  "np.max(batch_points_int) = ", np.max(batch_points_int) )
						print(  "np.min(batch_points_int) = ", np.min(batch_points_int) )
						print(  "np.max(batch_points) = ", np.max(batch_points) )
						print(  "np.min(batch_points) = ", np.min(batch_points) )

						model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(model_out, [self.batch_size])
						input_model_float[batch_points_int[:,0],batch_points_int[:,1],batch_points_int[:,2]] = np.reshape(batch_values, [self.batch_size])

					img1 = np.clip(np.amax(input_model_float, axis=0)*256, 0,255).astype(np.uint8)
					cv2.imwrite(self.res_sample_dir+str(epoch)+"_in.png",img1)

					img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
					cv2.imwrite(self.res_sample_dir+str(epoch)+"_out.png",img1)

				
				if idx==numShapes-1  and epoch%5 == 4 :
					self.save(self.checkpoint_dir, epoch)




	def test(self, specEpoch, FLAGS ):

		
		InputFolder = FLAGS.test_input_dir
		outputDim = FLAGS.test_outputDim

		StepSize_1 =0.002
		StepSize_2 =0.0001
		ae_optim_align = tf.train.AdamOptimizer(   StepSize_1, beta1=0.5).minimize(self.fineTune_loss ,         var_list=self.trainingVariables_align )
		ae_optim_decoder = tf.train.AdamOptimizer( StepSize_2, beta1=0.5).minimize(self.fineTune_loss ,         var_list=self.trainingVariables_decoders )

		outputStepList = [0, 50, 75, 100 ]
		
		FTstr =  str(StepSize_1) + '-' + str(StepSize_2)
		
		self.sess.run(tf.global_variables_initializer())
		
		could_load, checkpoint_counter = self.load_fortest(self.checkpoint_dir, specEpoch)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		
		## check whether an  InputFolder is provided or not
		if len(InputFolder) > 1 :
			if not os.path.exists(InputFolder):
				print( InputFolder, "does not exist")
				exit()
			self.test_res_dir = InputFolder + '_' + 'JS-' + str(outputDim) + "-" + str(specEpoch) + '-' + FTstr+'/'
			
			print("take input test parts from: ", InputFolder)
			print("write test result to: ", self.test_res_dir)

		else:
			print("take input test parts from the array: allpartPts2048_ori")
			print("write test result to: ", self.test_res_dir)

			
		if not os.path.exists(  self.test_res_dir   ):
			os.makedirs(  self.test_res_dir  )

		
		dima = self.test_size
		multiplier = int(outputDim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier

		aux_x = np.zeros([dima,dima,dima],np.int32)
		aux_y = np.zeros([dima,dima,dima],np.int32)
		aux_z = np.zeros([dima,dima,dima],np.int32)
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					aux_x[i,j,k] = i*multiplier
					aux_y[i,j,k] = j*multiplier
					aux_z[i,j,k] = k*multiplier
		coords_int = np.zeros([multiplier3,dima,dima,dima,3],np.int)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					coords_int[i*multiplier2+j*multiplier+k,:,:,:,0] = aux_x+i
					coords_int[i*multiplier2+j*multiplier+k,:,:,:,1] = aux_y+j
					coords_int[i*multiplier2+j*multiplier+k,:,:,:,2] = aux_z+k
		

		coords =  intGridNodes_2_floatPoints(coords_int, outputDim )
		coords = np.reshape(coords,[multiplier3,self.batch_size,3])

		coords_int = np.reshape(coords_int,[multiplier3,self.batch_size,3])
		
		print("self.trainSize = ", self.trainSize )
		

		random.seed(  0 )

		
		extraList = []
		
		for t in list(  range( FLAGS.testStartID, FLAGS.testEndID  ) ) + extraList:

	
			if len(InputFolder) > 1 :
				inputPlyPath =  InputFolder + '/' + str(t) + '-output.ply'
				if not os.path.exists( inputPlyPath ):
					print( inputPlyPath, "doesn't exist!!!")
					continue

				namePath = InputFolder + '/' + str(t) + '/' + str(t) + '.name.txt'
				if not os.path.exists( namePath ):
					print( namePath, "doesn't exist!!!")
					continue
				
			if t==0 or FLAGS.FTSteps>0:
				
				could_load, checkpoint_counter = self.load_fortest(self.checkpoint_dir, specEpoch)
				if could_load:
					print(" [*] Load SUCCESS")
				else:
					print(" [!] Load failed...")
					return


				## assign intial values to part_translates, and part_scales
				
				translate_zeros = tf.constant(0.0, tf.float32, shape=[self.num_of_parts, 2 ])
				scale_ones      = tf.constant(1.0, tf.float32, shape=[self.num_of_parts, 3 ])

				self.sess.run( self.part_translates.assign( translate_zeros ) )
				self.sess.run( self.part_scales.assign( scale_ones ) )

			
			dxb = t

			

			inputPC_list = []
			inputPC_normal_list = []
			inputPC_distance_list = []
			pointExistMask_list = []
			for lb in range( 1, self.num_of_parts+1  ):

				if len(InputFolder) > 1 :
					inputPC, inputPC_normal = load_ply_withnorml( InputFolder + '/' + str(t) + '-output.ply' )
					inputPC 		= inputPC[ ( lb-1)*2048 : lb*2048 ]
					inputPC_normal = inputPC_normal[ ( lb-1)*2048 : lb*2048 ]

					inputPC_list.append( inputPC )
					inputPC_normal_list.append(   inputPC_normal   )
					pointExistMask_list.append(   np.ones([2048], np.float32) * float( doesPartExist(inputPC) )   )

					#load namelist
					combList = loadObjNameList( InputFolder + '/' + str(t) + '/' + str(t) + '.name.txt' )				
					assert( len(combList) == self.num_of_parts )
					
					objIdx = self.objlist.index( combList[lb-1] )
					inputPC_distance_list.append( self.allpartPts2048_ori_dis[objIdx, ( lb-1)*2048 : lb*2048 ] )


				else:

					if t >= self.allpartPts2048_ori.shape[0]:
						exit(0)
						
					inputPC = self.allpartPts2048_ori[t, ( lb-1)*2048 : lb*2048 , : ]
					inputPC_normal = self.allpartPts2048_ori_normal[t, ( lb-1)*2048 : lb*2048 , : ]
					inputPC_list.append( inputPC )
					inputPC_normal_list.append(   inputPC_normal   )
					pointExistMask_list.append(   np.ones([2048], np.float32) * float( doesPartExist(inputPC) )   )
					
					inputPC_distance_list.append( self.allpartPts2048_ori_dis[t, ( lb-1)*2048 : lb*2048 ] )
					



			partTrans_list = []
			partScales_list = []
			for optstep in range(FLAGS.FTSteps+1):

                
				if optstep % 2 == 0:
					ae_optim = ae_optim_align
					print("ae_optim_align")
				else:
					ae_optim = ae_optim_decoder
					print("ae_optim_decoder")
	
				if optstep in outputStepList:
					model_float_wholeshape = np.zeros([outputDim+2, outputDim+2, outputDim+2],np.float32)
					for i in range(multiplier):
						for j in range(multiplier):
							for k in range(multiplier):
								minib = i*multiplier2+j*multiplier+k
								model_out_wholeshape = self.sess.run( self.sG,  
									feed_dict={								
										self.point_in: np.vstack( inputPC_list ),
										self.normal_in: np.vstack( inputPC_normal_list ),
										self.distance_in:  np.hstack(inputPC_distance_list),
										self.point_coord: coords[minib],
										self.point_exist_mask: np.hstack( pointExistMask_list )
									})
                                
								model_float_wholeshape[aux_x+i+1,aux_y+j+1,aux_z+k+1] = np.reshape(model_out_wholeshape, [dima,dima,dima])


					if FLAGS.outputHdf5:
						hdf5_path =  self.test_res_dir+ str(t) +".out." + str(optstep) + ".hdf5"
						hdf5_file = h5py.File(hdf5_path, 'w')
						joints = ( model_float_wholeshape[1:-1, 1:-1, 1:-1] > 0.5 ).astype( np.uint8 )
						hdf5_file.create_dataset("joints",           data=joints,          compression=4)
						hdf5_file.close()

					vertices, triangles = mcubes.marching_cubes(model_float_wholeshape, 0.5)
					vertices =  intGridNodes_2_floatPoints(vertices-1.0, outputDim )
					print("vertices.shape = ", vertices.shape )
					print("triangles.shape = ", triangles.shape )
					jointPlyPath = self.test_res_dir+ str(t) +".out." + str(optstep) + ".ply"
					write_ply_withfaces(jointPlyPath , vertices,  triangles)
					
					alignedPts, partTrans, partScales = self.sess.run(  [self.point_in_after_transform, self.part_translates_padded, self.part_scales ],
						feed_dict={
							self.point_in: np.vstack( inputPC_list ),
							self.normal_in: np.vstack( inputPC_normal_list ),
							self.distance_in:  np.hstack(inputPC_distance_list),
							self.point_exist_mask: np.hstack( pointExistMask_list )
						})

					partTrans_list.append( partTrans )
					partScales_list.append( partScales )
										
					write_colored_ply( self.test_res_dir+  str(t)  +".aligned_input." + str(optstep) + ".ply",   alignedPts,  np.vstack( inputPC_normal_list),    self.allpartPts2048_labels )

							
					# transform meshes
					if len(InputFolder) > 1 :
						targetFolder = self.test_res_dir +str(t) + '_' + str(optstep)
						copy_tree( InputFolder+"/" +str(t), targetFolder )

						meshPathList = []
						for pid in range( self.num_of_parts):

							mesh_path = targetFolder + '/' + self.part_name_list[pid] + '.ply'
							meshPathList.append(  mesh_path)
							print(mesh_path)
							if os.path.exists(  mesh_path ):
								vertices, tri_data = load_ply_withfaces( mesh_path )
								vertices = np.multiply( vertices,  partScales_list[-1][pid] ) + partTrans_list[-1][pid] 
								write_ply_withfaces( mesh_path, vertices,  tri_data)

						mergeMeshFiles(meshPathList,  self.test_res_dir+  str(t)  +".aligned_mesh." + str(optstep) + ".ply")

						mergeMeshFiles(meshPathList + [ jointPlyPath ] ,  self.test_res_dir+  str(t)  +".merged." + str(optstep) + ".ply")


				## do not run alignment optimization for test with GT alignment
				if optstep % 2 == 0 and  len(InputFolder) < 1 :
					continue

				_, matchLoss = self.sess.run([ae_optim, self.match_loss_beforeWeight ],
					feed_dict={
						self.point_in: np.vstack( inputPC_list ),
						self.normal_in: np.vstack( inputPC_normal_list ),
						self.distance_in:   np.hstack(inputPC_distance_list),
						self.point_exist_mask: np.hstack( pointExistMask_list )
					})

				print("opt step " + str(optstep) + ': ' , matchLoss)

			
			#### when "diffshapes" is in InputFolder, it means the folder contains alignment results where parts are from different shapes
			####  otherwise, it means the aligned parts are from the same object, then we have GT
			if "diffshapes" not in InputFolder:
				GT_voxels = np.copy( self.data_voxels_128[t] )

				assert(  GT_voxels.shape[0] ==  outputDim)

				vertices, triangles = mcubes.marching_cubes( np.squeeze( GT_voxels )*1.0 , 0.5 )
				vertices =  intGridNodes_2_floatPoints(vertices-1.0, outputDim )
				write_ply_withfaces( self.test_res_dir+  str(t)  +".gt.ply", vertices,  triangles)


			write_colored_ply( self.test_res_dir+  str(t)  +".input.ply",   np.vstack(inputPC_list),    np.vstack(inputPC_normal_list),    self.allpartPts2048_labels )



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

			print("###################### print_tensors_in_checkpoint_file:")			
			print_tensors_in_checkpoint_file(file_name=os.path.join(checkpoint_dir, ckpt_name), tensor_name='', all_tensors='')

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
