#!/usr/bin/env python3

import csv
from cv2 import imread, flip, cvtColor, COLOR_BGR2HSV
import numpy as np
import random
from keras.models import Model, Sequential
from keras.layers import (
    Dense, Dropout, Flatten,
    SpatialDropout2D)
from keras.layers.convolutional import (
     Convolution2D, MaxPooling2D)
from keras.optimizers import  SGD
from keras.regularizers import l2
from keras import metrics
from keras import backend as K

def apply_jitter(value, max_jitter=0.05):
  return value - max_jitter + max_jitter * 2 * random.random()

def load_data(sample_dir):
  samples = []
  with open(sample_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      samples.append(line)
  return samples

def load_and_optimize(path):
  image = imread(path)
  return image

def generate_batch(samples, sample_dir, batch_size=32):
  from sklearn.utils import shuffle
  while 1:
    images = []
    steering = []
    for line in shuffle(samples):
      sep='\\'
      img_center = load_and_optimize(sample_dir + '/IMG/' + line[0].split(sep)[-1])
      #img_left = load_and_optimize(sample_dir + '/IMG/' + line[1].split(sep)[-1])
      #img_right = load_and_optimize(sample_dir + '/IMG/' + line[2].split(sep)[-1])
      angle = np.array([float(line[3]),float(line[4])])

      #angle = apply_jitter(angle)

      images.append(img_center)
      #images.append(flip(img_center, 1))
      #images.append(img_left)
      #images.append(img_right)

      steering.append(angle)
      #steering.append(-angle)
      #steering.append(min(1.0, angle + 0.3))
      #steering.append(max(-1.0, angle - 0.3))

      if (len(images) >= batch_size):
        yield shuffle(np.array(images), np.array(steering))
        images = []
        steering = []

import keras

def create_model():
  #taken from https://github.com/emef/sdc
  input_shape = (160, 320, 3)
  use_adadelta = True,
  learning_rate = 0.01,
  W_l2 = 0.0001,
  scale = 16
  model = Sequential()
  model.add(Convolution2D(16, 5, 5,
                          input_shape=input_shape,
                          init="he_normal",
                          activation='relu',
                          border_mode='same'))
  model.add(SpatialDropout2D(0.1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(20, 5, 5,
                          init="he_normal",
                          activation='relu',
                          border_mode='same'))
  model.add(SpatialDropout2D(0.1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(40, 3, 3,
                          init="he_normal",
                          activation='relu',
                          border_mode='same'))
  model.add(SpatialDropout2D(0.1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(60, 3, 3,
                          init="he_normal",
                          activation='relu',
                          border_mode='same'))
  model.add(SpatialDropout2D(0.1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(80, 2, 2,
                          init="he_normal",
                          activation='relu',
                          border_mode='same'))
  model.add(SpatialDropout2D(0.1))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(128, 2, 2,
                          init="he_normal",
                          activation='relu',
                          border_mode='same'))
  model.add(Flatten())
  model.add(Dropout(0.5))
  model.add(Dense(
    output_dim=2,
    init='he_normal',
    W_regularizer=l2(W_l2)))

  optimizer = ('adadelta' if use_adadelta
               else SGD(lr=learning_rate, momentum=0.9))


  model.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    metrics=['rmse'])
  return model
def rmse(y_true, y_pred):
    '''Calculates RMSE
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def top_2(y_true, y_pred):
    return K.mean(tf.nn.in_top_k(y_pred, K.argmax(y_true, axis=-1), 2))

metrics.rmse = rmse
metrics.top_2 = top_2

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Agent training')
  parser.add_argument('--load', type=str, required=False, default=None, help='filename to load model from')
  parser.add_argument('--data', type=str, required=False, default='data', help='path to training data')
  parser.add_argument('--epochs', type=int, required=False, default=10, help='number of epochs')
  args = parser.parse_args()

  model = None
  if args.load:
    print("Loading model from ", args.load)
    model = keras.models.load_model(args.load)
  else:
    print("creating new model")
    model = create_model()

  data = 'data'
  if args.data:
    data = args.data

  print("Loading training data from", data)
  samples = load_data(data)
  print('Loaded {} samples'.format(len(samples)))
  train_samples = samples
  print(samples)
  train_generator = generate_batch(train_samples, data, batch_size=32)
  #for elem in train_generator:
  #  print(elem)
  try:
    model.fit_generator(train_generator, len(train_samples) * 4 / 32, epochs = args.epochs)
  except Exception as e:
    pass
  finally:
    model.save('model.h5')