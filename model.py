import os
import csv
from sklearn.model_selection import train_test_split
# Loads in InceptionV3
from keras.applications.inception_v3 import InceptionV3
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Input, Lambda, Flatten, Cropping2D
import tensorflow as tf

def read_the_data():
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_sample, validation_samples

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './IMG/'+batch_sample[0].split('/')[-1]
                left_name = './IMG/'+batch_sample[1].split('/')[-1]
                right_name = './IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def get_model_arcitechture():
    ch, row, col = 3, 160, 320  # Trimmed image format
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    input_size = 299
    model.add(Lambda(lambda x: x/127.5 - 1.,\
            input_shape=(col, row, ch),\
            output_shape=(col, row, ch)))
    model.add(Cropping2D(cropping=((70,25), (11,10))))
    model.add(Lambda(lambda image: tf.image.resize_images(image, (input_size, input_size))))
    model.add(InceptionV3(weights='imagenet', include_top=False,\
                        input_shape=(input_size,input_size,3)))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model
                          
                          

batch_size=32
# compile and train the model using the generator function
train_samples, validation_samples = read_the_data()
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
model = get_model_architecture()
model.fit_generator(train_generator, \
            steps_per_epoch=ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)
model.save('model.h5') 