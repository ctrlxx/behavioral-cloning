import csv
import cv2
import numpy as np

import pandas as pd
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, MaxPooling2D, Activation
from keras.layers import Cropping2D
from keras.optimizers import Adam

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if line[0] != 'center':
            lines.append(line)

images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        current_path = './data/' + source_path.strip()
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        correction = 0.2  # this is a parameter to tune
        steering_left = measurement + correction
        steering_right = measurement - correction
        if i == 1:
            measurement = steering_left
        elif i == 2:
            measurement = steering_right
        measurements.append(measurement)

print('training data ', len(images), len(measurements))

# -----------------------------------------------------------------------
def crop(image, top_percent, bottom_percent):

    assert 0 <= top_percent < 0.5, 'top_percent should be between 0.0 and 0.5'
    assert 0 <= bottom_percent < 0.5, 'top_percent should be between 0.0 and 0.5'

    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))

    return image[top:bottom, :]

def random_shear(image, steering_angle, shear_range=200):

    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle

def random_flip(image, steering_angle, flipping_prob=0.5):

    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle

def random_gamma(image):

    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def resize(image, new_dim):

    return scipy.misc.imresize(image, new_dim)

def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):

    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = random_flip(image, steering_angle)

    image = random_gamma(image)

    image = resize(image, resize_dim)

    return image, steering_angle

# -----------------------------------------------------------------------
images = np.array(images)
measurements = np.array(measurements)

x_train = []
y_train = []

for image, measurement in zip(images, measurements):
    new_image, new_angle = generate_new_image(image, measurement)
    x_train.append(new_image)
    y_train.append(new_angle)

print('training data ', len(images), len(measurements))



model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer=Adam(1e-4))
model.fit(np.array(x_train), np.array(y_train), validation_split=0.2, shuffle=True, epochs=8)

model.save('model.h5')
