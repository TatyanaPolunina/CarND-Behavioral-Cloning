import os
import csv
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Lambda, Flatten, Cropping2D, Dense, Conv2D, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.utils import shuffle

#read the data info from csv file
def read_the_data():
    samples = []
    with open('./driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            samples.append(line)
    #split the data to train (80%) and validation (20%) set
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

#load batch samples data generaton
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            correction = 0.2
            for batch_sample in batch_samples:
                center_name = './IMG/'+batch_sample[0].split('/')[-1]
                #left_name = './IMG/'+batch_sample[1].split('/')[-1]
                #right_name = './IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)                
                if center_image is None:
                    print(center_name)
                    continue
                center_angle = float(batch_sample[3])
                #left_angle = center_angle + correction
                #right_angle = center_angle - correction
                #flip the image and add it to training set
                flipped_image = np.fliplr(center_image)
                flipped_angle = -center_angle
                images.append(center_image)
                images.append(flipped_image)
                angles.append(center_angle)
                angles.append(flipped_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def get_model_architecture():
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(160, 320, 3),\
            output_shape=(160, 320, 3)))
    #crop the unneeded data from the image
    model.add(Cropping2D(cropping=((60,20), (0,0))))
    #provide convolution layers
    model.add(Conv2D(3, (3, 3), strides=(1, 2), padding='same', input_shape=(80, 320, 3)))
    model.add(Conv2D(3, (3, 3), strides=(1, 2), padding='same', input_shape=(80, 160, 3)))
    #add VGG pretrained model
    vgg = VGG16(weights='imagenet', include_top=False,\
                       input_shape=(80, 80, 3))    
    #use already pretrained parameters for current model
    for layer in vgg.layers: 
        vgg.trainable = False
    model.add(vgg)   
    #add fully connected layer
    model.add(Dense(256))
    model.add(Flatten())
    #add layer to receive needed output
    model.add(Dense(1))    
    #compile model with adam optimized and mse error.
    model.compile('adam', 'mse', ['accuracy'])
    return model
                    
batch_size=32
# compile and train the model using the generator function
train_samples, validation_samples = read_the_data()
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
model = get_model_architecture()
#see the final model architecture
print(model.summary())
model.fit_generator(train_generator, \
            steps_per_epoch=np.ceil(len(train_samples)/batch_size), \
            validation_data=validation_generator, \
            validation_steps=np.ceil(len(validation_samples)/batch_size), \
            epochs=5, verbose=1)
model.save('model.h5')
