import os
import scipy.misc
import numpy as np
import tensorflow as tf
import h5py

from partAlign_model import partAlignModel

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam")
flags.DEFINE_string("dataset", "03001627_vox", "The name of dataset")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./03001627_Chair/03001627_sampling_erode0.05_256", "Root directory of dataset [data]")
flags.DEFINE_boolean("train", False, "True for training, False for testing")
flags.DEFINE_integer("num_of_parts", 4, "")
flags.DEFINE_integer("shapeBatchSize", 64, "")
flags.DEFINE_integer("reWei", 0, "whether or not to assign different weights to different points")
flags.DEFINE_integer("diffShape", 0, "")
flags.DEFINE_string("gpu", '0', "")
flags.DEFINE_boolean("debug", False, "")
flags.DEFINE_string("workdir", "./default", "")

FLAGS = flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

print(FLAGS)


def main(_):
	
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	with tf.Session(config=run_config) as sess:
		PAM = partAlignModel(
					sess,
					256,
					is_training = FLAGS.train,
					dataset_name=FLAGS.dataset,
					checkpoint_dir=FLAGS.checkpoint_dir,
					data_dir=FLAGS.data_dir,
					num_of_parts=FLAGS.num_of_parts,
					reWei=FLAGS.reWei,
					shapeBatchSize=FLAGS.shapeBatchSize,
					FLAGS=FLAGS )

		if FLAGS.train:
			PAM.train(learning_rate=FLAGS.learning_rate,
						beta1=FLAGS.beta1,
						for_debug=FLAGS.debug, 
						epochNum=FLAGS.epoch )
		else:
			PAM.test(specEpoch=FLAGS.epoch, differentShapes=FLAGS.diffShape)

if __name__ == '__main__':
	tf.app.run()
