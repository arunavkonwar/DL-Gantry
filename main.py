'''
https://hackernoon.com/learning-keras-by-implementing-vgg16-from-scratch-d036733f2d5
'''
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
	from keras.applications import VGG16
	from keras import models
	from keras import layers

	
	#vgg16_model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
	
	conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

	model = models.Sequential()
	model.add(conv_base)

	conv_base.trainable = True

	set_trainable = False
	for layer in conv_base.layers:
		if layer.name == 'block5_conv1':
			set_trainable = True
		if set_trainable:
			layer.trainable = True
		else:
			layer.trainable = False
	
	
	model.add(layers.Flatten())
	'''
	model.add(layers.Dense(256, activation='relu'))
	layer.trainable = True
	model.add(layers.Dense(2, activation=None))
	layer.trainable = True
	'''
	model.add(layers.Dense(4096, activation='relu'))
	layer.trainable = True
	model.add(layers.Dense(4096, activation='relu'))
	layer.trainable = True
	model.add(layers.Dense(2, activation=None))
	layer.trainable = True
	
	'''
	vgg16_model = VGG16(weights='imagenet',
                  include_top=True,
                  input_shape=(224, 224, 3))
	model = Sequential()
	for layer in vgg16_model.layers:
    		model.add(layer)
		
	model.layers.pop()
	model.add(layers.Dense(2, activation=None))
	
	for layer in model.layers:
    		layer.trainable = True
	'''
	model.summary()
	print "length of the network:"
	print len(model.layers)
	return model
	




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
	import time
	from keras.callbacks import ModelCheckpoint

	np.random.seed(7) # for reproducibility

	batch_size = 14

	model = vgg16()
	#model.load_weights('/local/akonwar/trained_weights/trained_model_works_full_vgg_april21.h5')
	
	y_filename ='/udd/akonwar/code/deep-learning-for-visual-servoing/data/data_8k.txt'
	
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])
	y_data_train = y_data[:]
	#########################################
	
	#for 8k images dataset
	h5f = h5py.File('/local/akonwar/image_data/images_in_h5_format_8k.h5','r')
	
	x_data_train = h5f['dataset_1'][:]
	
	h5f = h5py.File('/local/akonwar/image_data/validation_images_in_h5_format_8k.h5','r')
	x_data_valid = h5f['dataset_1'][:]
	
	y_filename ='/udd/akonwar/code/deep-learning-for-visual-servoing/data/validation_data_8k.txt'
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])
	y_data_valid = y_data[:]



	# ======================================================================                     
	# Configure the training process:
	print('Preparing training ...')

	sgd = SGD(lr=1e-6, momentum=0.9, decay=0.001, nesterov=False)	
	#adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
	model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
	
	#update
	'''
	filepath="best_model.hdf5"	
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint] 
	'''

	iter=500
	# Train:
	print('Start training ...')
	start = time.time()
	
	history = model.fit(x = x_data_train, y = y_data_train,
		  epochs=iter,
		  batch_size=batch_size, validation_data = ( x_data_valid, y_data_valid ), shuffle = True, verbose = 1)  
		  #By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
	#test mode
	#score = model.evaluate(x=x_data_train, y=y_data_train, batch_size=50, verbose=1, sample_weight=None, steps=None)
	
	#for test mode
	'''
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	'''
	
	end = time.time()
	print ("Model took %0.2f seconds to train"%(end - start))
	
	print(history.history.keys()) 
	
	# summarize history for accuracy 
	plt.figure(1)  

	plt.subplot(211)  
	plt.plot(history.history['acc'])  
	plt.plot(history.history['val_acc'])  
	plt.title('model accuracy')  
	plt.ylabel('accuracy')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'validation'], loc='upper left')  

	# summarize history for loss  

	plt.subplot(212)  
	plt.plot(history.history['loss'])  
	plt.plot(history.history['val_loss'])  
	plt.title('model loss')  
	plt.ylabel('loss')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'validation'], loc='upper left')  
	#plt.show()
	plt.savefig('visualization_works_full_vgg_april21_1-500.png')


	model.save_weights('/local/akonwar/trained_weights/trained_model_works_full_vgg_april21_1-500.h5')
	

