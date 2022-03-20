import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Lambda, Input, Concatenate
from sklearn.model_selection import train_test_split
from cnn_model import cnn_model
from loss_and_accuracy import euclidean_distance, eucl_dist_output_shape, contrastive_accuracy, contrastive_loss, triplet_accuracy, triplet_loss
from tools import generate, get_sets, test_evaluation

'''
Functions included in the file:
    load_dataset
    siamese_network
    train_and_test_siamese_network
'''


def load_dataset():
  '''
  Loads the omniglot dataset from tfds and then split it to 80% training and 20% test.
  From the training dataset, it is split again into 80% training and 20% validation.

  test datasets for evaluation:

  1. Training split
  2. Test split
  3. Training and Test split combined (whole dataset)


  @ return

    Five different datasets for training, validation and 3 different test datasets.
  '''

  # Since there are 50 alphabet classes in total, 40 were used for training and 10 for testing.
  TEST_START_CLASS = 41 

  ds,ds_info = tfds.load('omniglot', split='train+test',with_info=True)

  X_combined = np.stack([tf.image.rgb_to_grayscale(char['image']).numpy() for char in ds]) # rgb_to_greysacle: Convert (105,105,3) shape to (105,105,1).
  y_combined = np.stack([char['alphabet'].numpy() for char in ds])

  # Split dataset by indexes
  train_indexes = np.where(y_combined < TEST_START_CLASS)
  test_indexes = np.where(y_combined >= TEST_START_CLASS)

  X = []
  y = []
  X_test = []
  y_test = []

  for index in train_indexes:
    X.append(X_combined[index])
    y.append(y_combined[index])

  for index in test_indexes:
    X_test.append(X_combined[index])
    y_test.append(y_combined[index])

  X = X[0]
  y = y[0]
  X_test = X_test[0]
  y_test = y_test[0]

  # Training split was split again into training and validation data with ration of 8:2.
  X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=1)
  
  train_dataset = (X_train, y_train)
  val_dataset = (X_val, y_val)
  dataset_1 = (X, y)
  dataset_2 = (X_test, y_test)
  dataset_3 = (X_combined, y_combined)    

  return train_dataset, val_dataset, dataset_1, dataset_2, dataset_3


def siamese_network(set_type, all_dataset): 
  '''
  Creates and trains Siamese network.

  @ param set_type:
    For selecting loss function.
    if "pair", creates a network using contrastive loss function,
    if "triplet", creates a network using triplet loss function.
  @ param all_dataset:
    A numpy array containing training dataset and validation dataset

  @ return:
    A trained siamese network model with its training history.
  '''
    
  train, val = all_dataset
  input_shape = (105,105,1)
  
  # Different number of epochs and batch_size used for loss functions
  epochs = 15
  verbose = 1
  batch_size = 256

  if set_type == "triplet":
    epochs = 20
    batch_size = 128

  # Create neural network model
  model = cnn_model()

  # Create input layers
  x_in_1 = Input(input_shape)
  x_in_2 = Input(input_shape)
  x_in_3 = Input(input_shape)

  # Create identical sub-networks that shares weights
  branch_1 = model(x_in_1)
  branch_2 = model(x_in_2)
  branch_3 = model(x_in_3)

  # Different layer depending on the set_type
  if set_type == "pair":
    x_out = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([branch_1, branch_2])
    siamese_model = tf.keras.Model([x_in_1, x_in_2], x_out)

  else:
    x_out = Concatenate()([branch_1, branch_2, branch_3])
    siamese_model = tf.keras.Model([x_in_1, x_in_2, x_in_3], x_out)
  
  optimizer = keras.optimizers.Adam(lr=0.001)
  loss = contrastive_loss
  metrics = contrastive_accuracy(0.5)

  if set_type == "triplet":
    loss = triplet_loss
    metrics = triplet_accuracy(1.0)

  siamese_model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

  history = siamese_model.fit(generate(batch_size, train, set_type),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=batch_size,
            validation_steps=batch_size,
            verbose=verbose,
            validation_data=(generate(batch_size, val, set_type))
            )
  
  
  return history, siamese_model



def train_and_test_siamese_network(set_type):
  '''
  Loads dataset, split dataset, create, train and evaluate Siamese network

  @ param set_type:
    "pair" or "triplet"
  '''
  train,val,test_1,test_2,test_3 = load_dataset()
  train_dataset = [train,val]
  test_datasets = [test_1, test_2, test_3]

  history, model = siamese_network(set_type, train_dataset)

  test_evaluation(model, history, test_datasets, set_type)
  

if __name__ == '__main__':
    a,b,c,d,e = load_dataset()
    x,y = get_sets(32, a, "triplet")
    print(x.shape, y.shape)
    # train_and_test_siamese_network("triplet")

