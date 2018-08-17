import sklearn
from sklearn.model_selection import train_test_split
import keras
import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.regularizers import l2, activity_l2


samples = []

with open('track1_normal/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('track1_normal_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
      
with open('track1_recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        

with open('track1_recovery_2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32, mirror =True):
    num_samples = len(samples)
    
    if mirror:
        batch_size = int(batch_size/2)
    
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+ batch_size]
            
            images = []
            angles = []
            
            for batch_sample in batch_samples:
                name = batch_sample[0]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                if mirror:
                    images.append(cv2.flip(center_image, 1))
                    angles.append(center_angle*-1)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=64, mirror=True)
validation_generator = generator(validation_samples, batch_size=64, mirror=True)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, activation='relu', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Convolution2D(36,5,5, activation='relu', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Convolution2D(48,5,5, activation='relu', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Convolution2D(64,3,3, activation='relu', subsample=(1, 1), W_regularizer=l2(0.001)))
model.add(Convolution2D(64,3,3, activation='relu', subsample=(1, 1), W_regularizer=l2(0.001)))
model.add(Flatten())
model.add(Dense(100, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(50, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(10, activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch = len(train_samples)*2,
                   validation_data = validation_generator, nb_val_samples = len(validation_samples)*2,
                   nb_epoch=5)


model.save('model.h5')

print('Model Saved!')
