#!/usr/bin/env python3

import csv
from cv2 import imread, flip
import numpy as np

def load_data(sample_dir):
  images = []
  steering = []
  with open(sample_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      filename = sample_dir + '/IMG/' + line[0].split('/')[-1]
      images.append(imread(filename))
      images.append(flip(images[-1], 1))
      steering.append(float(line[3]))
      steering.append(-steering[-1])
  return (np.array(images), np.array(steering))

images, steering_angles = load_data('data')
print('Loaded {} samples'.format(len(steering_angles)))

import keras

model = keras.models.Sequential()
model.add(keras.layers.convolutional.Cropping2D(cropping=((80, 20), (0, 0)), input_shape=images[0].shape))
model.add(keras.layers.core.Lambda(lambda x: x / 255.0 + 0.5))
model.add(keras.layers.Conv2D(6, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(16, (5, 5), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(30))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(images, steering_angles, validation_split=0.2, epochs=5, shuffle=True)
model.save('model.h5')
