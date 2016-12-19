# Import necessary libraries
import numpy as np
import pandas as pd
import os
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.misc import imresize
import cv2

print("All module loaded")

resizedX = 80
resizedY = 27
resizedCrop = 13
resizedOY = 40
# Get steering angles for Clock-wise driving, 5 laps
angles = pd.read_csv('data/driving_log.csv',header=None)
# angles.columns = ('Center Image','Left Image','Right Image','Steering Angle','Throttle','Brake','Speed')
angles.columns = ('center','left','right','steering','throttle','brake','speed')
# angles = np.array(angles['Steering Angle'])
angles = np.array(angles['steering'])
angles = angles[1:]
angles = angles.astype(np.float)



# Construct arrays for center, right and left images of controlled driving
images = np.asarray(os.listdir("data/IMG/"))
images = images[0:]
center = np.ndarray(shape=(len(angles), resizedY, resizedX, 3))
right = np.ndarray(shape=(len(angles), resizedY, resizedX, 3))
left = np.ndarray(shape=(len(angles), resizedY, resizedX, 3))

# Images are resized to 32x64 to increase training speeds
# Then we cropped top 12 pixels as they had no useful information for training
# Final size is 20 x 64 x 3
count = 0
for image in images:
    image_file = os.path.join('data/IMG', image)
    if image.startswith('center'):
        image_data = mpimg.imread(image_file)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2YUV)
        #Histogram the Y Channel to make it more robust
#         image_data[:,:,0] = cv2.equalizeHist(image_data[:,:,0])
        center[count % len(angles)] = imresize(image_data, (resizedOY,resizedX,3))[resizedCrop:,:,:]
    elif image.startswith('right'):
        image_data = mpimg.imread(image_file)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2YUV)
#         image_data[:,:,0] = cv2.equalizeHist(image_data[:,:,0])
        #Histogram the Y Channel to make it more robust
        right[count % len(angles)] = imresize(image_data, (resizedOY,resizedX,3))[resizedCrop:,:,:]
    elif image.startswith('left'):
        image_data = mpimg.imread(image_file)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2YUV)
#         image_data[:,:,0] = cv2.equalizeHist(image_data[:,:,0])
        #Histogram the Y Channel to make it more robust
        left[count % len(angles)] = imresize(image_data, (resizedOY,resizedX,3))[resizedCrop:,:,:]
    count += 1

%matplotlib inline
plt.imshow(center[0])

print(len(center))
print(len(angles))

# Concatenate all arrays in to combined training dataset and labels
# for left and right cameras, we add one more 0.1 angles to make harder left or right turn for training
X_train = np.concatenate((center, right, left), axis=0)
y_train = np.concatenate((angles, (angles - .2), (angles + .2)),axis=0)

# Split data set to create training/validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.05)

# Model
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(resizedY,resizedX,3)))
model.add(BatchNormalization(axis=1, input_shape=(resizedY,resizedX,3)))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(.2))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

# Adam optimizer and learning rate of .0001
adam = Adam(lr=0.0001)
model.compile(loss='mse',
              optimizer=adam)

# Save weights on each epoch
checkpoint = ModelCheckpoint(filepath = 'model.h5', verbose = 1, save_best_only=True, monitor='val_loss')

# Stop training while validation loss fails to decrease
callback = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

# # Train model for 20 epochs and a batch size of 128
model.fit(X_train,
        y_train,
        nb_epoch=20,
        verbose=1,
        batch_size=128,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, callback])

#print("Weights Saved")
json_string = model.to_json()
with open('model.json', 'w') as jsonfile:
    json.dump(json_string, jsonfile)
print("Model Saved")