#!/usr/bin/env python3

import csv
from cv2 import imread, flip, cvtColor, COLOR_BGR2HSV
import numpy as np
import random

def apply_jitter(value, max_jitter=0.1):
  return value - max_jitter + max_jitter * 2 * random.random()

def load_data(sample_dir):
  samples = []
  with open(sample_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      samples.append(line)
  return samples

def generate_batch(samples, sample_dir):
  from sklearn.utils import shuffle
  while 1:
    for line in shuffle(samples):
      images = []
      steering = []
      img_center = imread(sample_dir + '/IMG/' + line[0].split('/')[-1])
      img_center = cvtColor(img_center, COLOR_BGR2HSV)
      img_left = imread(sample_dir + '/IMG/' + line[1].split('/')[-1])
      img_left = cvtColor(img_left, COLOR_BGR2HSV)
      img_right = imread(sample_dir + '/IMG/' + line[2].split('/')[-1])
      img_right = cvtColor(img_right, COLOR_BGR2HSV)
      angle = apply_jitter(float(line[3]))

      images.append(img_center)
      images.append(flip(img_center, 1))
      images.append(img_left)
      images.append(img_right)

      steering.append(angle)
      steering.append(-angle)
      steering.append(angle + 0.6)
      steering.append(angle - 0.6)
      yield (np.array(images), np.array(steering))

import keras

def create_model():
  model = keras.models.Sequential()
  model.add(keras.layers.convolutional.Cropping2D(cropping=((80, 20), (0, 0)), input_shape=(160,320,3)))
  model.add(keras.layers.Lambda(lambda x: x / 255.0 - 0.5))
  model.add(keras.layers.Conv2D(10, (5, 5), activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
  model.add(keras.layers.Conv2D(20, (5, 5), activation='relu'))
  model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
  model.add(keras.layers.Conv2D(26, (5, 5), activation='relu'))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Conv2D(28, (5, 5), activation='relu'))
  model.add(keras.layers.Dropout(0.5))

  model.add(keras.layers.Flatten())

  model.add(keras.layers.Dense(30))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(20))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(1))

  model.compile(optimizer='adam', loss='mse')
  return model

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Agent training')
  parser.add_argument('--load', type=str, required=False, default=None, help='filename to load model from')
  parser.add_argument('--data', type=str, required=False, default='data', help='path to training data')
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

  from sklearn.model_selection import train_test_split
  train_samples, validation_samples = train_test_split(samples, test_size=0.2)
  train_generator = generate_batch(train_samples, data)
  validation_generator = generate_batch(validation_samples, data)

  model.fit_generator(train_generator, len(train_samples), validation_data=validation_generator, validation_steps = len(validation_samples), epochs = 3)
  model.save('model.h5')
