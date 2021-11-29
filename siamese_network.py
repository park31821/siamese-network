'''
This Assignment is done by:
    Yeon Jae Park (n10886249)
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Concatenate,Dropout, Activation, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
Functions included in the file:
    load_dataset
    euclidean_distance
    eucl_dist_output_shape
    contrastive_loss
    contrastive_accuracy
    triplet_loss
    tripelt_accuracy
    get_sets
    generate
    cnn_model
    siamese_network
    convert_to_similarity
    evaluate_prediction
    test_evaluation
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


def euclidean_distance(vects):
  '''
  @ param vects:
    vector represenatations of two images [x,y]

  @ return:
    euclidean distance metrics between two images  
  '''
  x, y = vects
  sum_square = tf.reduce_sum(tf.math.square(x - y), axis=1)
  return tf.sqrt(tf.maximum(sum_square, 1e-16))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred, margin=1.0):
  '''
  Calculates contrastive loss between two images.

  @ param y_true:
    true label (1 or 0)
  @ param y_pred:
    euclidean distance between two images to be compared
  @ param margin:
    margin for contrastive loss (default = 1.0)

  @ return:
    calculated loss value of two images
  '''
  y_true = tf.cast(y_true, dtype=tf.float32)
  square_pred = tf.math.square(y_pred)
  margin_square = tf.math.square(tf.maximum(margin - y_pred, 0.0))

  # If y_true = 0, only (1 - y_true) * margin_square calculated,
  # Else y_true * square_pred

  return 0.5* tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


def contrastive_accuracy(margin):
  '''
  Accuracy metrics used for contrastive loss function
  '''
  def acc(y_true, y_pred):
      y_true = tf.cast(y_true, dtype=tf.float32)
      return tf.reduce_mean(tf.cast(tf.equal(y_true, tf.cast(y_pred < margin, dtype=tf.float32)), dtype=tf.float32))
  return acc


def triplet_loss(y_true, y_pred, margin=0.7):
  '''
  Calculates triplet loss between two images.
  
  @ param y_true:
    label of the anchor image (not used in this function)
  @ param y_pred:
    An array with anchor, positive and negative images concatnated.
    First third of the array corresponds to anchor images,
    Second third corresopnds to positive images and
    Last third corresponds to negative images
  @ param margin:
    margin for triplet loss (default = 1.0)

  @ return:
    calculated loss value
  '''

  total_length = y_pred.shape.as_list()[-1]
  anchors = y_pred[:,0:int(total_length/3)]
  positives = y_pred[:,int(total_length/3):int(total_length*2/3)]
  negatives = y_pred[:,int(total_length*2/3):]

  #Distance between the anchor and the positive
  pos_dist = euclidean_distance([anchors, positives])
  #Distance between the anchor and the negative
  neg_dist = euclidean_distance([anchors, negatives])
  return tf.reduce_mean(tf.maximum((pos_dist - neg_dist) + margin, 0.0))


def triplet_accuracy(margin):
  '''
  Accuracy metrics used for triplet loss function
  '''

  def acc(y_true, y_pred):
      total_length = y_pred.shape.as_list()[-1]
      anchors = y_pred[:,0:int(total_length/3)]
      positives = y_pred[:,int(total_length/3):int(total_length*2/3)]
      negatives = y_pred[:,int(total_length*2/3):]

      #Distance between the anchor and the positive
      pos_dist = euclidean_distance([anchors, positives])
      #Distance between the anchor and the negative
      neg_dist = euclidean_distance([anchors, negatives])
      return tf.reduce_mean(tf.cast(tf.math.logical_and(tf.math.less_equal(pos_dist, neg_dist), tf.math.less_equal(pos_dist, margin)), dtype=tf.float32))
  return acc


def get_sets(batch_size, dataset, return_type):
  '''
  Radomly generates pairs or triplets for training and testing.

  @ param batch_size:
    number batch size to be generated in int. batch size number of pairs/triplets will be generated.
  @ param dataset:
    A dataset for generating paris/triplets in numpy array format.
  @ parm return_type:
    final return type in string. "pair" or "triplet"

  @ return:
    If return_type is "pair",
    returns two separate numpy arrays for batch_size number of image pairs and their label in binary value (0 or 1)
    half of the batch_size is positive pairs and the other half is negative pairs (if batch_size = 128, 64 positive pairs and 64 negative pairs)

    If return_type is "triplet",
    returns two separate numpy arrays for batch_size number of image triplets in form of [anchor, positive, negative] and label of anchor image (this is not used for training).
  '''
    
  TRAIN_CLASS_LENGTH = 25870 # 80% of the total length of the whole dataset (train + test).
  img_per_line = 2 # Number of indexes to be created for numpy array
  
  X_dataset, y_dataset = dataset
  number_samples, w, h, l = X_dataset.shape
  unique_classes = np.unique(y_dataset) # unique alphabet classes in the dataset
  categories = np.random.choice(unique_classes, size=(batch_size,), replace=True) # radomly select alphabet classes for generating data.
  
  if return_type == 'triplet':
      img_per_line = 3 # if triplet, we need 3 different arrays
  
  if len(unique_classes) > TRAIN_CLASS_LENGTH: # This is for generating image pairs/triplets from combined dataset
      train_classes = np.asarray(np.where(unique_classes <= TRAIN_CLASS_LENGTH))[0]
      test_classes = np.asarray(np.where(unique_classes > TRAIN_CLASS_LENGTH))[0]
      
      train_categories = np.random.choice(train_classes, size=(batch_size//2,), replace=False)
      test_categories = np.random.choice(test_classes, size=(batch_size//2,), replace=False)
      categories = np.concatenate((train_categories, test_categories))

  
  sets = [np.zeros((batch_size, w, h, l)) for i in range(img_per_line)] # Create sepatate arrays for images
  
  targets = np.zeros((batch_size,)) # Array for target label

  if return_type == "pair":
    targets[batch_size//2:] = 1 # Half positive pairs, half negative pairs
  
  for x in range(batch_size):
      category = categories[x] # Choose an alphabet class from randomly chosen classes. 
      category_positive = category
      samples = np.where(y_dataset == category)[0] # Get indexes of characters in the category
      first_sample = np.random.randint(0,len(samples)) 

      labels_confirm = [y_dataset[samples[first_sample]], 0]

      sets[0][x,:,:,:] = X_dataset[samples[first_sample]] # Radomly choose an image from the category and add to the array.
      

      # Randomly choose negative image 
      if (x < batch_size // 2 and return_type == 'pair') or return_type == 'triplet': 
          while True:
              category_2 = np.random.choice(unique_classes)
          
              if category_2 != category:
                  category = category_2
                  break

      samples = np.where(y_dataset == category)[0]
      second_sample = np.random.randint(0,len(samples))
      labels_confirm[1] = y_dataset[samples[second_sample]]
      sets[-1][x,:,:,:] = X_dataset[samples[second_sample]]

      if return_type == 'triplet':
          # Add positive image to the array 
          samples = np.where(y_dataset == category_positive)[0]
          sample = np.random.randint(0,len(samples))
          sets[1][x,:,:,:] = X_dataset[samples[sample]]
          targets[x] = category_positive
      
  return np.array(sets), np.array(targets)


def generate(batch_size, dataset, return_type):
  '''
  A generator to generate random image pairs/triplets for every epoch.

  The parameters have same definiton as get_sets()

  @ yield:
    yields return array and label from get_sets()
  '''

  while True:
    sets, targets = get_sets(batch_size, dataset, return_type)

    if return_type == "pair":
      yield [sets[0], sets[1]], targets

    else:
      yield [sets[0], sets[1], sets[2]], targets


def cnn_model():
  '''
  Returns a neural network model to be used for Siamese Netowork.

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


def siamese_network(set_type, all_dataset): 
  '''
  Creates and trains Siamese network.

  @ param set_type:
    For selecting loss function.
    if "pair", creates a network using contrastive loss function,
    if "triplet", creates a network using triplet loss function.
  @ param all_dataset:
    A numpy array containing training dataset and validataion dataset

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


def convert_to_similarity(a, b):
  '''
  Converts y_pred from triplet loss Siamese network to similarity between 0 and 1.

  @ param a, b:
    y_pred arrays to calculate the similarity
  @ return:
    similarity between 0 and 1
  '''

  nominator = np.dot(a, b)   
  a_norm = np.sqrt(np.sum(a**2))
  b_norm = np.sqrt(np.sum(b**2))
  
  denominator = a_norm * b_norm
  similarity = nominator / denominator
  
  return similarity


def evaluate_prediction(y_true, y_pred, set_type):
  '''
  Evaluates the prediction made by the trained model.
  Prints confusion matrix and accuracy of the model.

  @ param y_true:
    true label - a numpy array of binary values
  @ param y_pred:
    y_pred obtained from the model prediction in an array format
  @ param set_type:
    "pair" for contrastive loss function model
    "triplet for triplet loss function model
  '''
    
  # The shape of y_pred of triplet loss model is (128,96), which anchors, positives and negatives are concatenated.
  anchor_index = 32
  positive_index = 64
  triplet_batch_size = 128
  pred = (y_pred.ravel() < 0.5).astype('float32') # Converts y_pred to binary values

  if set_type == "triplet":
    true_label = np.ones(triplet_batch_size)
    false_label = np.zeros(triplet_batch_size)

    y_true = np.concatenate((true_label, false_label))

    label_true = []
    label_false = []

    for x in y_pred:
      # Converts y_pred of triplet loss model to similarity value and then to binary value for comparison with y_true.
      similarity_positive = float(convert_to_similarity(x[:anchor_index], x[anchor_index:positive_index]) > 0.5) 
      similarity_negative = float(convert_to_similarity(x[:anchor_index], x[positive_index:]) > 0.5)

      label_true.append(similarity_positive)
      label_false.append(similarity_negative)

    pred = np.concatenate((np.asarray(label_true), np.asarray(label_false)))

  # Prints out confusion matrix and accuracy of the model
  print("Confusion Matrix: ")
  print(tf.math.confusion_matrix(y_true, pred))
  print(" ")
  accuracy = np.mean(pred == y_true)

  print(f'Prediction Accuracy: {accuracy*100}%\n')


def test_evaluation(model, history, test_datasets, set_type):

  '''
  Evaluates the trained Siamese network.

  @ param model:
    a trained model
  @ param history:
    training history of the model
  @ param test_datasets:
    an numpy array containing all test datasets
  @ param set_type:
    "pair" or "triplet"

  @ return:
    Prints graphs of model accuracy and loss,
    Evaluates the model using the test dataset, which prints confusion matrix and accuracy.
  '''

  batch_size = 256

  if set_type == "triplet":
    batch_size = 128

  Test_number = 1

  # Plot model accuracy graph
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()
  print(" ")

  # Plot model loss graph
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()
  print(" ")

  # Evaluates the model for each test dataset
  for test in test_datasets:
    print(f'Evaluation with Test Datset {Test_number}: \n')
    pairs, targets = get_sets(batch_size, test, set_type)
    if set_type == 'pair':
      y_pred = model.predict([pairs[0], pairs[1]])
    else:
      y_pred = model.predict([pairs[0], pairs[1], pairs[2]])
    evaluate_prediction(targets, y_pred, set_type)
    Test_number += 1


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
    train_and_test_siamese_network("pair")
    # train_and_test_siamese_network("triplet")

