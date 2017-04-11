#!/usr/bin/env python3

import csv
from cv2 import imread, flip, cvtColor, COLOR_BGR2HSV
import numpy as np
import random

def apply_jitter(values, max_jitter=0.25):
  for i in range(len(values)):
    values[i] = values[i] - max_jitter + max_jitter * 2 * random.random()
  return values

def load_data(sample_dir):
  images = []
  steering = []
  with open(sample_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      img_center = imread(sample_dir + '/IMG/' + line[0].split('/')[-1])
      img_center = cvtColor(img_center, COLOR_BGR2HSV)
      img_left = imread(sample_dir + '/IMG/' + line[1].split('/')[-1])
      img_left = cvtColor(img_left, COLOR_BGR2HSV)
      img_right = imread(sample_dir + '/IMG/' + line[2].split('/')[-1])
      img_right = cvtColor(img_right, COLOR_BGR2HSV)
      angle = float(line[3])

      images.append(img_center)
      images.append(flip(img_center, 1))
      images.append(img_left)
      images.append(img_right)

      steering.append(angle)
      steering.append(-angle)
      steering.append(angle + 0.6)
      steering.append(angle - 0.6)
  return (np.array(images), np.array(steering))

images, steering_angles = load_data('data')
print('Loaded {} samples'.format(len(steering_angles)))

import keras

model = keras.models.Sequential()
model.add(keras.layers.convolutional.Cropping2D(cropping=((80, 20), (0, 0)), input_shape=images[0].shape))
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
model.fit(images, apply_jitter(steering_angles), validation_split=0.2, epochs=5, shuffle=True)
model.save('model.h5')
