
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

import tflearn
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.contrib.framework.python.framework import checkpoint_utils

from latent_3d_points_decoder import decoder_with_fc_only
from latent_3d_points.structural_losses.tf_approxmatch import approx_match, match_cost

class PAE(object):
	def __init__(self, sess, pointSampleDim,  FLAGS, is_training = False,  \
					dataset_name='default', checkpoint_dir=None, data_dir='./data', rdDisplaceRange=0.0,  rdScaleRange=0.1, \
					partLabel=1, loadDataset=True,  \
					shapeBatchSize=16, ptsBatchSize=16384 ):


		self.workdir =  os.path.split( data_dir )[0]
		if self.workdir[:2] == "./":
			self.workdir = self.workdir[2:]


		print("self.workdir = ", self.workdir)

		self.sess = sess

		self.ptsSampDim = pointSampleDim			
		
		self.input_dim = 128

		self.dataset_name = dataset_name
		self.checkpoint_dir = self.workdir + '/' + checkpoint_dir + "_ae_part_" + str(partLabel)
		self.res_sample_dir = self.checkpoint_dir + "/res_sample/"
		self.test_res_dir = self.workdir + '/' +"test_res_part_" + str(partLabel) + "/"
		if FLAGS.FeedforwardTrainSet:
			self.test_res_dir = self.workdir + '/' +"test_res_part_" + str(partLabel) + "_on_train_set/"

		self.data_dir = data_dir
		self.rdDisplaceRange = rdDisplaceRange
		self.rdScaleRange = rdScaleRange
		self.partLabel = int(partLabel)
		self.shapeBatchSize = shapeBatchSize
		self.ptsBatchSize = ptsBatchSize
		self.batch_size = self.ptsBatchSize

		if not os.path.exists(  self.workdir   ):
			os.makedirs(  self.workdir  )
			
		if not os.path.exists(  self.checkpoint_dir   ):
			os.makedirs(  self.checkpoint_dir  )

		if not os.path.exists(  self.res_sample_dir   ):
			os.makedirs(  self.res_sample_dir  )

		if not os.path.exists(  self.test_res_dir   ):
			os.makedirs(  self.test_res_dir  )

			
		############ load dataset only when   loadDataset==True
		if os.path.exists(self.data_dir+'/'+self.dataset_name+'.hdf5')   and loadDataset:  
			self.data_dict = h5py.File(self.data_dir+  "/" + self.dataset_name+'.hdf5', 'r')
			
			part_points_shape = self.data_dict['part_points_'+str(self.ptsSampDim) + '_masked'][:, :, :self.ptsBatchSize, :].shape

			self.trainSize = int( part_points_shape[0] * 0.8 )	
			trainSize = self.trainSize 	
			
			self.num_of_parts = part_points_shape[1]	
				
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


				self.allpartPts2048_ori          = self.data_dict['allpartPts2048_ori_sorted'][:trainSize]
				self.allpartPts2048_ori_normal   = self.data_dict['allpartPts2048_ori_normal_sorted'][:trainSize]

				self.singlePartPoints2048        = self.allpartPts2048_ori[       : , (self.partLabel-1)*2048 : self.partLabel*2048 ]
				self.singlePartPoints2048_normal = self.allpartPts2048_ori_normal[: , (self.partLabel-1)*2048 : self.partLabel*2048 ]


				
				### load point clouds before erosion
				self.data_dict_noEr = h5py.File(self.data_dir.split("_erode")[0] + "_erode0.0_256_ptsSorted/" + self.dataset_name+'.hdf5', 'r')
				self.allpartPts2048_ori_noEr          = self.data_dict_noEr['allpartPts2048_ori_sorted'][:trainSize]
				self.allpartPts2048_ori_normal_noEr   = self.data_dict_noEr['allpartPts2048_ori_normal_sorted'][:trainSize]
				self.data_dict_noEr.close()


				self.singlePartPoints2048_noEr        = self.allpartPts2048_ori_noEr[       : , (self.partLabel-1)*2048 : self.partLabel*2048 ]
				self.singlePartPoints2048_normal_noEr = self.allpartPts2048_ori_normal_noEr[: , (self.partLabel-1)*2048 : self.partLabel*2048 ]

					
				self.objlist = self.objlist[:trainSize]

				self.data_examples_dir =  self.checkpoint_dir + "/train_examples_10/"

			else:

				self.shapeBatchSize = 1
			

				if FLAGS.FeedforwardTrainSet:
					self.allpartPts2048_ori          = self.data_dict['allpartPts2048_ori_sorted'][:trainSize]
					self.allpartPts2048_ori_normal   = self.data_dict['allpartPts2048_ori_normal_sorted'][:trainSize]
				else:
					self.allpartPts2048_ori          = self.data_dict['allpartPts2048_ori_sorted'][trainSize:]
					self.allpartPts2048_ori_normal   = self.data_dict['allpartPts2048_ori_normal_sorted'][trainSize:]

				self.singlePartPoints2048        = self.allpartPts2048_ori[       : , (self.partLabel-1)*2048 : self.partLabel*2048 ]
				self.singlePartPoints2048_normal = self.allpartPts2048_ori_normal[: , (self.partLabel-1)*2048 : self.partLabel*2048 ]


				### load point clouds before erosion
				self.data_dict_noEr = h5py.File(self.data_dir.split("_erode")[0] + "_erode0.0_256_ptsSorted/" + self.dataset_name+'.hdf5', 'r')
				if FLAGS.FeedforwardTrainSet:
					self.allpartPts2048_ori_noEr          = self.data_dict_noEr['allpartPts2048_ori_sorted'][:trainSize]
					self.allpartPts2048_ori_normal_noEr   = self.data_dict_noEr['allpartPts2048_ori_normal_sorted'][:trainSize]
				else:
					self.allpartPts2048_ori_noEr          = self.data_dict_noEr['allpartPts2048_ori_sorted'][trainSize:]
					self.allpartPts2048_ori_normal_noEr   = self.data_dict_noEr['allpartPts2048_ori_normal_sorted'][trainSize:]

				self.data_dict_noEr.close()

				self.singlePartPoints2048_noEr        = self.allpartPts2048_ori_noEr[       : , (self.partLabel-1)*2048 : self.partLabel*2048 ]
				self.singlePartPoints2048_normal_noEr = self.allpartPts2048_ori_normal_noEr[: , (self.partLabel-1)*2048 : self.partLabel*2048 ]

				
				if FLAGS.FeedforwardTrainSet:
					self.objlist = self.objlist[:trainSize]
				else:
					self.objlist = self.objlist[trainSize:]
				

				self.data_examples_dir = self.checkpoint_dir + "/test_examples_10/"


			print(  "self.allpartPts2048_ori.shape = ", self.allpartPts2048_ori.shape  )
			print(  "self.singlePartPoints2048.shape = ", self.singlePartPoints2048.shape  )
			outputExamples = 1
			if outputExamples:
				
				if not os.path.exists(  self.data_examples_dir   ):
					os.makedirs(  self.data_examples_dir  )
					
				for t in range(5):
					print(self.objlist[t])

					lables = np.ones(  2048*self.num_of_parts, dtype=np.uint8 )
					for lb in range(1, self.num_of_parts+1):
						lables[(lb-1)*2048 : lb*2048 ] = lb
					self.allpartPts2048_labels = lables		

					write_colored_ply( self.data_examples_dir + self.objlist[t]+"_ori.ply",     self.allpartPts2048_ori[t],  self.allpartPts2048_ori_normal[t], self.allpartPts2048_labels )
					output_point_cloud_ply( self.data_examples_dir + self.objlist[t]+"_part.ply",   self.singlePartPoints2048[t]  )

				
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

			self.batch_size = self.test_size*self.test_size*self.test_size
			
		
		self.build_model()
		
		tflearn.is_training(is_training)

		

	def build_model(self):
		
		
		self.inputPC        = tf.placeholder( shape=[self.shapeBatchSize, 2048,   3], dtype=tf.float32 )
		self.inputPC_normal = tf.placeholder( shape=[self.shapeBatchSize, 2048,   3], dtype=tf.float32 )
		self.inputPC_gt     = tf.placeholder( shape=[self.shapeBatchSize, 2048,   3], dtype=tf.float32 )
			
		self.inputPC_pt_norm = tf.concat(  [self.inputPC, self.inputPC_normal] ,  axis=2  )
		


		self.scope_predix="part_"+str(self.partLabel)

		self.E   = get_pointNet_code_of_singlePart( self.inputPC_pt_norm,  None,       scope_predix=self.scope_predix,  is_training=True, reuse=False)
		self.G   = self.pc_generator_2048( self.E,     scope_predix=self.scope_predix,  phase_train=True, reuse=False)

		self.sE  = get_pointNet_code_of_singlePart( self.inputPC_pt_norm,  None,       scope_predix=self.scope_predix,  is_training=False, reuse=True)
		self.sG  = self.pc_generator_2048( self.sE,     scope_predix=self.scope_predix,  phase_train=False, reuse=True)


		match = approx_match(self.G, self.inputPC_gt)
		self.loss = tf.reduce_mean(match_cost(self.G, self.inputPC_gt, match))
	
		smatch = approx_match(self.sG, self.inputPC_gt)
		self.sloss = tf.reduce_mean(match_cost(self.sG, self.inputPC_gt, smatch))



		self.trainingVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= '.*'+self.scope_predix  )
		print("################ trainingVariables: " )
		for v in self.trainingVariables:
			print(v.name)
		
		self.saver = tf.train.Saver(max_to_keep=1000, var_list=self.trainingVariables )
		
				
	def pc_generator_2048(self,  z,   scope_predix="part_1", phase_train=True, reuse=False):
		with tf.variable_scope(scope_predix+"_generator_2048") as scope:
			if reuse:
				scope.reuse_variables()			

			output = decoder_with_fc_only( latent_signal=z, layer_sizes=[ 256, 256, 2048*3], b_norm=False, b_norm_finish=False, verbose=False )

			return tf.reshape(output, [self.shapeBatchSize, 2048,3])
	



	def train(self,  learning_rate, beta1, for_debug, epochNum ):


		ae_optim_single = tf.train.AdamOptimizer( learning_rate, beta1=beta1).minimize( self.loss )
		self.sess.run(tf.global_variables_initializer())
		
		
		batch_index_list = np.arange(  self.allpartPts2048_ori.shape[0]  )
		
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
		numShapes = numShapes - numShapes % self.shapeBatchSize			

		print("batch_index_list = ", batch_index_list)
		print("numShapes = ", numShapes)


		if for_debug:
			numShapes = 1
			batch_index_list = np.arange(numShapes)


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

			ae_optim = None
			if epoch < 200:
				ae_optim = ae_optim_single
				print( "ae_optim = ae_optim_single")
			else:
				ae_optim = ae_optim_single
				print( "ae_optim = ae_optim_single")


			np.random.shuffle(batch_index_list)
			avg_loss = 0
			avg_loss_new = 0
			avg_num = 0
			ignoreCount = 0
			for idx in range(0, numShapes,  self.shapeBatchSize):

				dxb = batch_index_list[ idx:idx+self.shapeBatchSize ]
				
				
				batch_inputPC = np.copy( self.singlePartPoints2048[ dxb ] )
				batch_inputPC_normal = np.copy( self.singlePartPoints2048_normal[ dxb ] )
				batch_inputPC_gt = np.copy( self.singlePartPoints2048_noEr[ dxb ] )


				### random transform !!!!
				batch_inputPC_gt, batch_inputPC  =  randomDisplaceScale_TwoPointSet( batch_inputPC_gt, batch_inputPC, \
																			displaceRange=self.rdDisplaceRange, scaleRange=self.rdScaleRange ) 

				# Update AE network
				_, errAE = self.sess.run([ae_optim, self.loss ],
					feed_dict={
						self.inputPC:  batch_inputPC,
						self.inputPC_normal:   batch_inputPC_normal,
						self.inputPC_gt:   batch_inputPC_gt
					})
				avg_loss += errAE
				avg_num += 1
				if (idx%16 == 0):
					print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
					print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, avgloss: %.8f" % (epoch, epochNum, idx, numShapes, time.time() - start_time, avg_loss/avg_num ))

				if idx==(numShapes-self.shapeBatchSize):
					
					if epoch%5 == 4 :
						self.save(self.checkpoint_dir, epoch)

						
					dxb = batch_index_list[ idx:idx+self.shapeBatchSize ]
						
					batch_inputPC = np.copy( self.singlePartPoints2048[ dxb ] )
					batch_inputPC_normal = np.copy( self.singlePartPoints2048_normal[ dxb ] )
					batch_inputPC_gt = np.copy( self.singlePartPoints2048_noEr[ dxb ] )

				
					if not doesPartExist(  batch_inputPC[0] ):
						continue
					

					### random transform !!!!
					batch_inputPC_gt, batch_inputPC  =  randomDisplaceScale_TwoPointSet( batch_inputPC_gt, batch_inputPC, \
																				displaceRange=self.rdDisplaceRange, scaleRange=self.rdScaleRange ) 

					
					model_out = self.sess.run(self.sG,
						feed_dict={
						self.inputPC:  batch_inputPC,
						self.inputPC_normal:   batch_inputPC_normal
						})
					
					write_colored_ply( self.res_sample_dir+str(epoch)+"_in.ply",   batch_inputPC[0], batch_inputPC_normal[0], self.allpartPts2048_labels )
					output_point_cloud_ply( self.res_sample_dir+str(epoch)+"_out.ply",   model_out[0]  )


					print("[sample]")
			
			print("ignoreCount = ", ignoreCount)

				

	def get_latent_code( self,  input_PC, input_PC_normal ):

		if len( input_PC.shape ) == 2:
			input_PC = np.expand_dims( input_PC, 0 )
			input_PC_normal = np.expand_dims( input_PC_normal, 0 )

		## assume checkpoint has beeen loaded
		latent_code = self.sess.run( self.sE,  
			feed_dict={
				self.inputPC:  input_PC,
				self.inputPC_normal:  input_PC_normal
			})
		return latent_code


	def test(self, specEpoch ):
		
		could_load, checkpoint_counter = self.load_fortest(self.checkpoint_dir, specEpoch)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		if not os.path.exists(  self.test_res_dir + 'epoch_' + str(specEpoch) + '/'  ):
			os.makedirs(  self.test_res_dir  + 'epoch_' + str(specEpoch) + '/' )


		random.seed(  0 )
		for t in range(  100 ):

			if t >= self.singlePartPoints2048.shape[0]:
				break

			batch_inputPC = np.copy( self.singlePartPoints2048[ t:t+1 ] )
			batch_inputPC_normal = np.copy( self.singlePartPoints2048_normal[ t:t+1 ] )
					
			### random transform !!!!
			batch_inputPC  =  randomDisplaceScale_PointSet( batch_inputPC,  \
														displaceRange=self.rdDisplaceRange, scaleRange=self.rdScaleRange ) 

			if not doesPartExist(  batch_inputPC[0] ):
				continue

			model_out = self.sess.run(self.sG,
				feed_dict={
				self.inputPC:  batch_inputPC,
				self.inputPC_normal:   batch_inputPC_normal
				})
			

			write_colored_ply( self.test_res_dir  + 'epoch_' + str(specEpoch) + '/'+  str(t)  +".input.ply",   batch_inputPC[0], batch_inputPC_normal[0], self.allpartPts2048_labels )
			output_point_cloud_ply( self.test_res_dir  + 'epoch_' + str(specEpoch) + '/'+str(t)+".output.ply",   model_out[0]  )

			
			print("[sample]")
	


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

		print( "checkpoint_dir = ", checkpoint_dir )
		print( "self.model_dir = ", self.model_dir )

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
