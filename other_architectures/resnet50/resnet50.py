def vgg16():
	import keras
	from keras.models import Sequential
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam
	from keras.metrics import categorical_crossentropy
	from keras.layers.normalization import BatchNormalization
	from keras.layers.convolutional import *
	import matplotlib.pyplot as plt
	from keras.utils import plot_model 


	vgg16_model = keras.applications.resnet50.ResNet50()
	'''
	model = Sequential()
	for layer in vgg16_model.layers:
		model.add(layer)
	
	#model.layers.pop()
	#model.layers.pop()
	#model.layers.pop()

	#model.add(Dense(2, activation=None))
	
	for layer in model.layers:
		layer.trainable = True

	#model.add(Dense(2, activation='linear'))
	
	model.summary()
	return model
	'''
	vgg16_model.summary()




if __name__ == "__main__":
	import os
	import numpy as np
	from keras import optimizers
	from keras.models import Sequential
	from keras.models import load_model
	from keras.layers import Activation
	from keras.layers.core import Dense, Flatten
	from keras.optimizers import Adam, SGD
	from keras.metrics import categorical_crossentropy
	#import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pyplot as plt
	import h5py
	from keras.utils import plot_model
	#from keras.callbacks import ModelCheckpoint
	#import utils
	#import models
	import time
	from keras.callbacks import ModelCheckpoint

	np.random.seed(7) # for reproducibility

	batch_size = 14

	#model = load_model('vgg16_edit.h5')
	model = vgg16()
	#model.load_weights('trained_model_weights_dense_trainable_sgd_pose.h5')
	
	
	
