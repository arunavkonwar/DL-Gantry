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

	weights_path=None
	
	input_shape = (224, 224, 3)

	model = Sequential([
	    Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
	    Conv2D(64, (3, 3), activation='relu', padding='same'),
	    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	    Conv2D(128, (3, 3), activation='relu', padding='same'),
	    Conv2D(128, (3, 3), activation='relu', padding='same',),
	    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	    Conv2D(256, (3, 3), activation='relu', padding='same',),
	    Conv2D(256, (3, 3), activation='relu', padding='same',),
	    Conv2D(256, (3, 3), activation='relu', padding='same',),
	    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
	    Flatten(),
	    Dense(2, activation='linear')
	])

	model.summary()

	if weights_path:
		model.load_weights(weights_path)

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
	model.load_weights('/local/akonwar/trained_weights/trained_model_other0_1-20.h5')
	
	y_filename ='/udd/akonwar/code/deep-learning-for-visual-servoing/data/data_8k.txt'
	
	#y_filename ='/udd/akonwar/code/deep-learning-for-visual-servoing/data/data_40k.txt'
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
	
	y_filename ='/udd/akonwar/code/deep-learning-for-visual-servoing/data/validation_data_8k.txt'
	y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1])
	y_data_valid = y_data[:]



	# ======================================================================                     
	# Configure the training process:
	print('Preparing training ...')
	#adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

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

	iter=20
	# Train:
	print('Start training ...')
	start = time.time()

	history = model.fit(x = x_data_train, y = y_data_train,
		  epochs=iter,
		  batch_size=batch_size, validation_data = ( x_data_valid, y_data_valid ), shuffle = False, verbose = 1)  
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
	plt.savefig('visualization_other0_21-40.png')


	model.save_weights('/local/akonwar/trained_weights/trained_model_other0_21-40.h5')
	#model.save('trained_model.h5')
	

