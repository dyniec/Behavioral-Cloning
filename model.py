#!/usr/bin/env python3

import csv
from cv2 import imread, flip, cvtColor, COLOR_BGR2HSV
import numpy as np
import random

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
  image = cvtColor(image, COLOR_BGR2HSV)
  if (random.random() > 0.5):
    image[:,:,2] = image[:,:,2] * .25+np.random.uniform()
  return image

def generate_batch(samples, sample_dir, batch_size=32):
  from sklearn.utils import shuffle
  while 1:
    images = []
    steering = []
    for line in shuffle(samples):
      img_center = load_and_optimize(sample_dir + '/IMG/' + line[0].split('/')[-1])
      img_left = load_and_optimize(sample_dir + '/IMG/' + line[1].split('/')[-1])
      img_right = load_and_optimize(sample_dir + '/IMG/' + line[2].split('/')[-1])
      angle = float(line[3])

      angle = apply_jitter(angle)

      images.append(img_center)
      images.append(flip(img_center, 1))
      images.append(img_left)
      images.append(img_right)

      steering.append(angle)
      steering.append(-angle)
      steering.append(min(1.0, angle + 0.3))
      steering.append(max(-1.0, angle - 0.3))

      if (len(images) >= batch_size):
        yield shuffle(np.array(images), np.array(steering))
        images = []
        steering = []

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

  model.add(keras.layers.Dense(30, activation='tanh'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(20, activation='tanh'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(10, activation='tanh'))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Dense(1))

  model.compile(optimizer='adam', loss='mse')
  return model

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
  train_generator = generate_batch(train_samples, data, batch_size=32)

  model.fit_generator(train_generator, len(train_samples) * 4 / 32, epochs = args.epochs)
  model.save('model.h5')
