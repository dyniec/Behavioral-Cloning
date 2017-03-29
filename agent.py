#!/usr/bin/env python3

import csv
from cv2 import imread

def load_data(sample_dir):
  images = []
  steering = []
  with open(sample_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      filename = sample_dir + '/IMG/' + line[0].split('/')[-1]
      images.append(imread(filename))
      steering.append(float(line[3]))
  return (images, steering)

images, steering_angle = load_data('data')
print('Loaded {} samples'.format(len(steering_angle)))
