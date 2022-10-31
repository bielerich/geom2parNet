##########################################################################
# MODELHANDLER.PY
# class returning tensorflow neural network model --geom2parNet--
##########################################################################
#!/usr/bin/python3
import sys
sys.path.insert(0, ".")
import CaseHandler as ch
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

##########################################################################
# define functions for building convolutional and dense layers
# using batch normalization

def conv_batchNorm(x,filters,kernel_size):
    initializer = tf.keras.initializers.RandomUniform(minval=-5,maxval=5)
    x = layers.Conv1D(  filters,  
                        kernel_size=kernel_size,
                        padding="same", 
                        kernel_initializer=initializer, 
        		)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("sigmoid")(x)

def dense_batchNorm(x,units):
    initializer = tf.keras.initializers.RandomUniform(minval=-5,maxval=5)
    x = layers.Dense(   units, 
                        kernel_initializer=initializer, 
        		)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    return layers.Activation("sigmoid")(x)

##########################################################################
# FEATURE TRANSFORMATION
# takes input data, constructs transformation matrix T and
# applies T directly upon input data
##########################################################################
# constructing mini neural network t-net according to pointNet structure
# inputData    = input data
# num_features = number of points 
def transformationNet(inputLayer, number_of_features):
    initializer = tf.keras.initializers.RandomUniform(minval=-1,maxval=1)   
    # initialize and add layers to transformationNet     
    x = conv_batchNorm(inputLayer,32,5)
    x = conv_batchNorm(x,64,5)
    x = conv_batchNorm(x,256,5)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_batchNorm(x,32)
    x = layers.Dense(	number_of_features * number_of_features,
        		activation='sigmoid',
        		bias_initializer='zeros',
    		)(x)
   # build transformation matrix T and apply multiplication
    T = layers.Reshape((number_of_features,number_of_features))(x)
    return layers.Dot(axes=(2,1))([inputLayer,T])

##########################################################################
# GEOM2PARNET CLASS
##########################################################################
# initialize class
class geom2parNet():
    def __init__(self,modeltype='geom2parNet'):
        self.CASE     	= ch.CaseHandler()
        # define loss function
        self.loss    	= tf.keras.losses.Huber()
        # define number of epochs (given by user)
        self.epochs    	= self.CASE.epochs
        # define optimizer and learning rate
        lr_schedule    	= keras.optimizers.schedules.ExponentialDecay(
                      		initial_learning_rate=0.1,
                          	decay_steps=self.CASE.number_of_passes,
                           	decay_rate=0.9
                           	)
        self.optimizer 	= tf.keras.optimizers.Adam(
        					learning_rate=lr_schedule
        					)

    # construct neural network model
    def initModel(self):
       # defining shape of input layer
       # number_of_features    = number of points
       # number_of_subfeatures = dimension of points, i.e. 3 
       # number_of_labels		= number of labels, i.e. number of 
       #                         parameters
        shape           = (self.CASE.number_of_features, 
                           self.CASE.number_of_subfeatures
                           )
        inputLayer  	= keras.Input(shape=shape)

        # construct layers
        x = transformationNet(inputLayer, 3)
        x = conv_batchNorm(x, 64, 1)
        x = conv_batchNorm(x, 128, 1)
        x = conv_batchNorm(x, 256, 1)        
        x = layers.GlobalMaxPooling1D()(x)        
        x = dense_batchNorm(x, 2048)
        x = dense_batchNorm(x, 1024)
        x = dense_batchNorm(x, 512)
        x = dense_batchNorm(x, 128)
        x = dense_batchNorm(x, 16)
        outputLayer	= layers.Dense(self.CASE.number_of_labels, 
                               activation="sigmoid", 
                               bias_initializer='zeros',
                               )(x)
        model   = keras.Model(inputs=inputLayer, 
        		      outputs=outputLayer, 
        		      name="geom2parNet"
        		      )

        return model


