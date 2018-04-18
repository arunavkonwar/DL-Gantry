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
	model.add(layers.Flatten())
	model.add(layers.Dense(2, activation='linear'))
	

	conv_base.trainable = True

	set_trainable = False
	for layer in conv_base.layers:
		if layer.name == 'block5_conv1':
			set_trainable = True
		if set_trainable:
			layer.trainable = True
		else:
			layer.trainable = False
	
	
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
	#from keras.callbacks import ModelCheckpoint
	#import utils
	#import models
	import time
	from keras.callbacks import ModelCheckpoint

	np.random.seed(7) # for reproducibility

	batch_size = 14

	#model = load_model('vgg16_edit.h5')
	model = vgg16()
	#model.load_weights('/local/akonwar/trained_weights/trained_model_vgg_quentin_values_1-20.h5')
	
	y_filename ='./data/data_8k.txt'
	
	#y_filename ='./data/data_40k.txt'
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])
	y_data_train = y_data[:]

	#########################################
	
	#for 8k images dataset
	h5f = h5py.File('/local/akonwar/image_data/images_in_h5_format_8k_uint8.h5','r')
	
	#for 40k images dataset
	#h5f = h5py.File('/local/akonwar/image_data/images_in_h5_format_40k.h5','r')
	x_data_train = h5f['dataset_1'][:]
	
	h5f = h5py.File('/local/akonwar/image_data/validation_images_8k_uint8.h5','r')
	x_data_valid = h5f['dataset_1'][:]
	
	y_filename ='./data/validation_data_8k.txt'
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])
	y_data_valid = y_data[:]



	# ======================================================================                     
	# Configure the training process:
	print('Preparing training ...')
	#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

	sgd = SGD(lr=1e-4, momentum=0.9, decay=0.00139, nesterov=True)	
	#adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	#model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
	model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
	
	#update
	'''
	filepath="best_model.hdf5"	
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint] 
	'''

	iter=30
	# Train:
	print('Start training ...')
	start = time.time()

	history = model.fit(x = x_data_train, y = y_data_train,
		  epochs=iter,
		  batch_size=batch_size, validation_data = ( x_data_valid, y_data_valid ), shuffle = True, verbose = 1)  
		  #By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
	
	#history = model.evaluate(x=x_data_train, y=y_data_train, batch_size=50, verbose=1, sample_weight=None, steps=None)
	
	end = time.time()
	print ("Model took %0.2f seconds to train"%(end - start))
	
	print(history.history.keys()) 
	#for test mode
	'''
	print('Test loss:', history[0])
	print('Test accuracy:', history[1])
	'''
	plt.figure(1)  

	# summarize history for accuracy 
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	
	epochs = range(len(acc))

	plt.subplot(211)  
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()  

	# summarize history for loss  

	plt.subplot(212)  
	
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	#plt.show()
	plt.savefig('visualization_fchollet_tut_noDense_sgd_posenet_LR_1-30.png')


	model.save_weights('/local/akonwar/trained_weights/trained_model_fchollet_tut_noDense_sgd_posenet_LR_1-30.h5')
	#model.save('trained_model.h5')
	

