import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


'''
Functions included in the file:
    get_sets
    generate
    convert_to_similarity
    evaluate_prediction
    test_evaluation
'''


def get_sets(batch_size, dataset, return_type):
  '''
  Randomly generates pairs or triplets for training and testing.

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