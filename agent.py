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
