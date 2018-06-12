# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000):
	
             
	input = Input(shape=(224, 224, 3))

	conv1 = Conv2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1')(input)

	pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='pool1')(conv1)

	norm1 = BatchNormalization(axis=3, name='norm1')(pool1)

	reduction2 = Conv2D(64,1,1,border_mode='same',activation='relu',name='reduction2')(norm1)

	conv2 = Conv2D(192,3,3,border_mode='same',activation='relu',name='conv2')(reduction2)

	norm2 = BatchNormalization(axis=3, name='norm2')(conv2)

	pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2')(norm2)

	icp1_reduction1 = Conv2D(96,1,1,border_mode='same',activation='relu',name='icp1_reduction1')(pool2)

	icp1_out1 = Conv2D(128,3,3,border_mode='same',activation='relu',name='icp1_out1')(icp1_reduction1)


	icp1_reduction2 = Conv2D(16,1,1,border_mode='same',activation='relu',name='icp1_reduction2')(pool2)

	icp1_out2 = Conv2D(32,5,5,border_mode='same',activation='relu',name='icp1_out2')(icp1_reduction2)


	icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp1_pool')(pool2)

	icp1_out3 = Conv2D(32,1,1,border_mode='same',activation='relu',name='icp1_out3')(icp1_pool)


	icp1_out0 = Conv2D(64,1,1,border_mode='same',activation='relu',name='icp1_out0')(pool2)


	icp2_in = merge([icp1_out0, icp1_out1, icp1_out2, icp1_out3],mode='concat',concat_axis=3,name='icp2_in')






	icp2_reduction1 = Conv2D(128,1,1,border_mode='same',activation='relu',name='icp2_reduction1')(icp2_in)

	icp2_out1 = Conv2D(192,3,3,border_mode='same',activation='relu',name='icp2_out1')(icp2_reduction1)


	icp2_reduction2 = Conv2D(32,1,1,border_mode='same',activation='relu',name='icp2_reduction2')(icp2_in)

	icp2_out2 = Conv2D(96,5,5,border_mode='same',activation='relu',name='icp2_out2')(icp2_reduction2)


	icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp2_pool')(icp2_in)

	icp2_out3 = Conv2D(64,1,1,border_mode='same',activation='relu',name='icp2_out3')(icp2_pool)


	icp2_out0 = Conv2D(128,1,1,border_mode='same',activation='relu',name='icp2_out0')(icp2_in)


	icp2_out = merge([icp2_out0, icp2_out1, icp2_out2, icp2_out3],mode='concat',concat_axis=3,name='icp2_out')






	icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='icp3_in')(icp2_out)

	icp3_reduction1 = Conv2D(96,1,1,border_mode='same',activation='relu',name='icp3_reduction1')(icp3_in)

	icp3_out1 = Conv2D(208,3,3,border_mode='same',activation='relu',name='icp3_out1')(icp3_reduction1)


	icp3_reduction2 = Conv2D(16,1,1,border_mode='same',activation='relu',name='icp3_reduction2')(icp3_in)

	icp3_out2 = Conv2D(48,5,5,border_mode='same',activation='relu',name='icp3_out2')(icp3_reduction2)


	icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp3_pool')(icp3_in)

	icp3_out3 = Conv2D(64,1,1,border_mode='same',activation='relu',name='icp3_out3')(icp3_pool)


	icp3_out0 = Conv2D(192,1,1,border_mode='same',activation='relu',name='icp3_out0')(icp3_in)


	icp3_out = merge([icp3_out0, icp3_out1, icp3_out2, icp3_out3],mode='concat',concat_axis=3,name='icp3_out')






	cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid',name='cls1_pool')(icp3_out)

	cls1_reduction_pose = Conv2D(128,1,1,border_mode='same',activation='relu',name='cls1_reduction_pose')(cls1_pool)


	cls1_fc1_flat = Flatten()(cls1_reduction_pose)

	cls1_fc1_pose = Dense(1024,activation='relu',name='cls1_fc1_pose')(cls1_fc1_flat)

	cls1_fc_pose_xyz = Dense(3,name='cls1_fc_pose_xyz')(cls1_fc1_pose)

	cls1_fc_pose_wpqr = Dense(4,name='cls1_fc_pose_wpqr')(cls1_fc1_pose)






	icp4_reduction1 = Conv2D(112,1,1,border_mode='same',activation='relu',name='icp4_reduction1')(icp3_out)

	icp4_out1 = Conv2D(224,3,3,border_mode='same',activation='relu',name='icp4_out1')(icp4_reduction1)


	icp4_reduction2 = Conv2D(24,1,1,border_mode='same',activation='relu',name='icp4_reduction2')(icp3_out)

	icp4_out2 = Conv2D(64,5,5,border_mode='same',activation='relu',name='icp4_out2')(icp4_reduction2)


	icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp4_pool')(icp3_out)

	icp4_out3 = Conv2D(64,1,1,border_mode='same',activation='relu',name='icp4_out3')(icp4_pool)


	icp4_out0 = Conv2D(160,1,1,border_mode='same',activation='relu',name='icp4_out0')(icp3_out)


	icp4_out = merge([icp4_out0, icp4_out1, icp4_out2, icp4_out3],mode='concat',concat_axis=3,name='icp4_out')






	icp5_reduction1 = Conv2D(128,1,1,border_mode='same',activation='relu',name='icp5_reduction1')(icp4_out)

	icp5_out1 = Conv2D(256,3,3,border_mode='same',activation='relu',name='icp5_out1')(icp5_reduction1)


	icp5_reduction2 = Conv2D(24,1,1,border_mode='same',activation='relu',name='icp5_reduction2')(icp4_out)

	icp5_out2 = Conv2D(64,5,5,border_mode='same',activation='relu',name='icp5_out2')(icp5_reduction2)


	icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp5_pool')(icp4_out)

	icp5_out3 = Conv2D(64,1,1,border_mode='same',activation='relu',name='icp5_out3')(icp5_pool)


	icp5_out0 = Conv2D(128,1,1,border_mode='same',activation='relu',name='icp5_out0')(icp4_out)


	icp5_out = merge([icp5_out0, icp5_out1, icp5_out2, icp5_out3],mode='concat',concat_axis=3,name='icp5_out')






	icp6_reduction1 = Conv2D(144,1,1,border_mode='same',activation='relu',name='icp6_reduction1')(icp5_out)

	icp6_out1 = Conv2D(288,3,3,border_mode='same',activation='relu',name='icp6_out1')(icp6_reduction1)


	icp6_reduction2 = Conv2D(32,1,1,border_mode='same',activation='relu',name='icp6_reduction2')(icp5_out)

	icp6_out2 = Conv2D(64,5,5,border_mode='same',activation='relu',name='icp6_out2')(icp6_reduction2)


	icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp6_pool')(icp5_out)

	icp6_out3 = Conv2D(64,1,1,border_mode='same',activation='relu',name='icp6_out3')(icp6_pool)


	icp6_out0 = Conv2D(112,1,1,border_mode='same',activation='relu',name='icp6_out0')(icp5_out)


	icp6_out = merge([icp6_out0, icp6_out1, icp6_out2, icp6_out3],mode='concat',concat_axis=3,name='icp6_out')






	cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid',name='cls2_pool')(icp6_out)

	cls2_reduction_pose = Conv2D(128,1,1,border_mode='same',activation='relu',name='cls2_reduction_pose')(cls2_pool)


	cls2_fc1_flat = Flatten()(cls2_reduction_pose)

	cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1')(cls2_fc1_flat)

	cls2_fc_pose_xyz = Dense(3,name='cls2_fc_pose_xyz')(cls2_fc1)

	cls2_fc_pose_wpqr = Dense(4,name='cls2_fc_pose_wpqr')(cls2_fc1)    






	icp7_reduction1 = Conv2D(160,1,1,border_mode='same',activation='relu',name='icp7_reduction1')(icp6_out)

	icp7_out1 = Conv2D(320,3,3,border_mode='same',activation='relu',name='icp7_out1')(icp7_reduction1)


	icp7_reduction2 = Conv2D(32,1,1,border_mode='same',activation='relu',name='icp7_reduction2')(icp6_out)

	icp7_out2 = Conv2D(128,5,5,border_mode='same',activation='relu',name='icp7_out2')(icp7_reduction2)


	icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp7_pool')(icp6_out)

	icp7_out3 = Conv2D(128,1,1,border_mode='same',activation='relu',name='icp7_out3')(icp7_pool)


	icp7_out0 = Conv2D(256,1,1,border_mode='same',activation='relu',name='icp7_out0')(icp6_out)


	icp7_out = merge([icp7_out0, icp7_out1, icp7_out2, icp7_out3],mode='concat',concat_axis=3,name='icp7_out')






	icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='icp8_in')(icp7_out)

	icp8_reduction1 = Conv2D(160,1,1,border_mode='same',activation='relu',name='icp8_reduction1')(icp8_in)

	icp8_out1 = Conv2D(320,3,3,border_mode='same',activation='relu',name='icp8_out1')(icp8_reduction1)


	icp8_reduction2 = Conv2D(32,1,1,border_mode='same',activation='relu',name='icp8_reduction2')(icp8_in)

	icp8_out2 = Conv2D(128,5,5,border_mode='same',activation='relu',name='icp8_out2')(icp8_reduction2)


	icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp8_pool')(icp8_in)

	icp8_out3 = Conv2D(128,1,1,border_mode='same',activation='relu',name='icp8_out3')(icp8_pool)


	icp8_out0 = Conv2D(256,1,1,border_mode='same',activation='relu',name='icp8_out0')(icp8_in)

	icp8_out = merge([icp8_out0, icp8_out1, icp8_out2, icp8_out3],mode='concat',concat_axis=3,name='icp8_out')






	icp9_reduction1 = Conv2D(192,1,1,border_mode='same',activation='relu',name='icp9_reduction1')(icp8_out)

	icp9_out1 = Conv2D(384,3,3,border_mode='same',activation='relu',name='icp9_out1')(icp9_reduction1)


	icp9_reduction2 = Conv2D(48,1,1,border_mode='same',activation='relu',name='icp9_reduction2')(icp8_out)

	icp9_out2 = Conv2D(128,5,5,border_mode='same',activation='relu',name='icp9_out2')(icp9_reduction2)


	icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp9_pool')(icp8_out)

	icp9_out3 = Conv2D(128,1,1,border_mode='same',activation='relu',name='icp9_out3')(icp9_pool)


	icp9_out0 = Conv2D(384,1,1,border_mode='same',activation='relu',name='icp9_out0')(icp8_out)

	icp9_out = merge([icp9_out0, icp9_out1, icp9_out2, icp9_out3],mode='concat',concat_axis=3,name='icp9_out')






	cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),border_mode='valid',name='cls3_pool')(icp9_out)

	cls3_fc1_flat = Flatten()(cls3_pool)

	cls3_fc1_pose = Dense(2048,activation='relu',name='cls3_fc1_pose')(cls3_fc1_flat)


	cls3_fc_pose_xyz = Dense(3,name='cls3_fc_pose_xyz')(cls3_fc1_pose)

	cls3_fc_pose_wpqr = Dense(4,name='cls3_fc_pose_wpqr')(cls3_fc1_pose)






	posenet = Model(input=input, output=[cls1_fc_pose_xyz, cls1_fc_pose_wpqr, cls2_fc_pose_xyz, cls2_fc_pose_wpqr, cls3_fc_pose_xyz, cls3_fc_pose_wpqr])


if __name__ == '__main__':
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

    from keras.utils import plot_model
    
    np.random.seed(7) # for reproducibility

    batch_size = 14

    model = ResNet50(include_top=False, weights='imagenet') 
    print(model.summary())
    # plot graph
    plot_model(model, to_file='multiple_inputs.png')



    y_filename ='/udd/akonwar/code/deep-learning-for-visual-servoing/data/data_4DOF.txt'
    
    y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1,2,5])
    y_data_train = y_data[:]
    #########################################
    
    #for 8k images dataset
    #h5f = h5py.File('/local/akonwar/image_data/images_in_h5_format_8k.h5','r')
    #h5f = h5py.File('/local/akonwar/image_data/images_in_h5_format_8k_by255.h5','r')
    h5f = h5py.File('/local/akonwar/image_data/4DOF.h5','r')
    
    x_data_train = h5f['dataset_1'][:]
    
    #h5f = h5py.File('/local/akonwar/image_data/validation_images_in_h5_format_8k.h5','r')
    h5f = h5py.File('/local/akonwar/image_data/4DOF.h5','r')
    x_data_valid = h5f['dataset_1'][:]
    
    y_filename ='/udd/akonwar/code/deep-learning-for-visual-servoing/data/data_4DOF.txt'
    y_data = np.loadtxt(y_filename, delimiter='  ', usecols=[0,1,2,5])
    y_data_valid = y_data[:]



    # ======================================================================                     
    # Configure the training process:
    print('Preparing training ...')

    #sgd = SGD(lr=1e-5, momentum=0.9, decay=0.00139, nesterov=True) 
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error')
    #model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    
    #update
    '''
    filepath="best_model.hdf5"  
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint] 
    '''

    iter=150
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
    plt.savefig('viz_resnet50_90percent_1-150_adam_0001_velocity_hd.png')


    model.save_weights('/local/akonwar/trained_weights/trained_model_resnet50_dual-loss_4DOF.h5')
