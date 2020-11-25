import os
import scipy.misc
import numpy as np

from partAE_model import PAE

import tensorflow as tf
import h5py

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_string("dataset", "03001627_vox", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./03001627_Chair/03001627_sampling_erode0.05_256", "Root directory of dataset [data]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("FeedforwardTrainSet", False, "feed Training Set as test input")

flags.DEFINE_integer("ptsBatchSize", 16384, "point samples batch size")

flags.DEFINE_integer("partLabel", 1, "")

flags.DEFINE_string("gpu", '0', "")
##
flags.DEFINE_boolean("debug", False, "")
###
flags.DEFINE_string("workdir", "./default", "")


FLAGS = flags.FLAGS


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

print(FLAGS)


def main(_):
	
	
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		pae = PAE(
				sess,
				256,
				FLAGS, 
				is_training = FLAGS.train,
				dataset_name=FLAGS.dataset,
				checkpoint_dir=FLAGS.checkpoint_dir,
				data_dir=FLAGS.data_dir,
				partLabel=FLAGS.partLabel,
				ptsBatchSize=FLAGS.ptsBatchSize )

		if FLAGS.train:
			pae.train(learning_rate=FLAGS.learning_rate,
						beta1=FLAGS.beta1,
						for_debug=FLAGS.debug, 
						epochNum=FLAGS.epoch )
		else:
			pae.test( specEpoch=FLAGS.epoch )
			
if __name__ == '__main__':
	tf.app.run()
