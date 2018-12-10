#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:45:42 2018

@author: abinash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Building the CNN
classifier = Sequential()
#Convolution
classifier.add(Convolution2D(filters=32, kernel_size=(3,3),
                             strides=(1,1), input_shape=(64,64, 3),
                             activation='relu'))
#MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a 2nd convolution layer
classifier.add(Convolution2D(filters=32, kernel_size=(3,3),
                             strides=(1,1),
                             activation='relu'))
#MaxPooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattenning
classifier.add(Flatten())

#Fully connected layer
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#Fitting the model
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('dataset/training_set',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(train_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)
