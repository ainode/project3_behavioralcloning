import os
import argparse
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
import cv2 
from keras import backend as K
import tensorflow as tf
import csv  
from PIL import Image
import numpy as np
import random

train_data_path = 'C:/Users/owner/Downloads/self driving car Udacity Nanodegree/behavioralCloning_p3/data/data'
label_data_path = 'C:/Users/owner/Downloads/self driving car Udacity Nanodegree/behavioralCloning_p3/data/data/driving_log.csv'

steering_data = []
with open(label_data_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:  
        steering_data.append(row)          

print ('Length of steering data: ', len(steering_data))

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
X_train, y_train = shuffle(steering_data, steering_data)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
                                                      y_train,
                                                      test_size=0.20,
                                                      random_state=42)

#shift the camera image horizontally and adjust the steering accordingly
def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = float(steer) + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(64,64))    
    return image_tr,steer_ang

#flip image horizontally by random to provide more diverse samples
def flip(image, y_steer):
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image = cv2.flip(image,1)
        y_steer = float(y_steer)  * -1
    return image, y_steer    

#generate batches from the file on the fly because of memory constrains    
def generate_arrays_from_file(steering_data, batch_size=128):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while 1:
        for i in range(batch_size):
            random = int(np.random.choice(len(steering_data),1))
            image = Image.open(train_data_path + '/' + steering_data[random]['center'])
            angle = steering_data[random]['steering']
            img = image.resize((64,64))   
            feature = np.array(img, dtype=np.float32)
            feature, angle = trans_image(feature, angle, 10.0)            
            feature, angle = flip(feature, angle)
            batch_train[i] = feature        
            batch_angle[i] = angle
        yield batch_train, batch_angle

#NVidia convolutional model
def get_model(time_len=1):

    model = Sequential()
    model.add(BatchNormalization(mode=1, input_shape=(64, 64, 3)))        
    model.add(Convolution2D(24, 5, 5, subsample= (2, 2), name='conv1_1'))
    #model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample= (2, 2), name='conv2_1'))
    #model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample= (2, 2), name='conv3_1'))
    #model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='conv4_1'))
    #model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample= (1, 1), name='conv4_2'))
    #model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1164, name = "dense_0"))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(100,  name = "dense_1"))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(50, name = "dense_2"))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(10, name = "dense_3"))
    model.add(Activation('relu'))
    model.add(Dense(1, name = "dense_4"))
    #model.add(Lambda(atan_layer, output_shape = atan_layer_shape, name = "atan_0"))
    model.compile(optimizer="adam", loss="mse")

    return model

model = get_model()
#generate data for training
train_gen = generate_arrays_from_file(X_train)
#generate data for validation
val_gen = generate_arrays_from_file(X_valid)

model.fit_generator(
    train_gen,
    samples_per_epoch=len(X_train),
    nb_epoch=10,verbose=1,
    validation_data= val_gen,
    nb_val_samples=len(X_valid)
  )
    
json_model = model.to_json()
open('C:\\Users\\owner\\CarND-Term1-Starter-Kit\\model.json', 'w').write(json_model)

model.save_weights('C:\\Users\\owner\\CarND-Term1-Starter-Kit\\model.h5')
    