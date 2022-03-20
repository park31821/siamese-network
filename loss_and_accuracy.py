import tensorflow as tf


'''
Functions included in the file:
    euclidean_distance
    eucl_dist_output_shape
    contrastive_loss
    contrastive_accuracy
    triplet_loss
    triplet_accuracy

'''


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
    An array with anchor, positive and negative images concatenated.
    First third of the array corresponds to anchor images,
    Second third correspond to positive images and
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
