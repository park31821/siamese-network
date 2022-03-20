import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Lambda, Dropout, Activation, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D


'''
Functions included in the file:
    cnn_model
'''

def cnn_model():
  '''
  Returns a neural network model to be used for Siamese Network.

  Implemented and modified from the model of https://www.kaggle.com/gawarek/one-shot-learning-and-triplet-loss?fbclid=IwAR0tvWKQJAmjMmBrTyB8zJgoOsWR5OwgECAYmJ3u7zqHLEnPqNX6Oja6In0
  '''

  model = Sequential()
  model.add(Lambda(lambda x: x / 255.0))
  model.add(Conv2D(64, (5, 5), padding='valid', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(ZeroPadding2D(padding=(1, 1)))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  model.add(Dropout(.25))
  
  model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))    
  model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(256, (5, 5), padding='valid', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  model.add(Dropout(.25))

  model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))    
  model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(256, (3, 3), padding='valid', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  model.add(Dropout(.25))
  
  model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Conv2D(512, (3, 3), padding='valid', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  model.add(Dropout(.1))

  
  model.add(GlobalAveragePooling2D())
  
  model.add(Dense(32, kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
  return model

