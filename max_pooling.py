import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

import tensorflow as tf
from pca import digits_pca

digits = pd.read_csv('data/train.csv')

y = digits["label"]
x = digits.drop(labels=["label"], axis=1)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
x_train = x_train / 255.0
x_train_final = x_train.values.reshape(-1, 28, 28, 1)
# x_train_final = x_train_reshape.astype("float32")/255

print("Train data size :" + str(len(x_train)))
print("Validation data size :" + str(len(x_val)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2, 2)))
model.add(Dropout(0, 25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0, 25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.summary()

datagen = ImageDataGenerator(zoom_range=0.1,height_shift_range=0.1,width_shift_range=0.1,rotation_range=10)

model.compile(loss='categorical_crossentropy',
optimizer=Adam(learning_rate=1e-4),
metrics=['accuracy'])
# set learning rate
annealer = LearningRateScheduler (lambda x:1e-3*0.9**x)# fit model
hist = model.fit(datagen.flow(x_train_final, y_train, batch_size=28),steps_per_epoch =500, epochs =10,
                           callbacks=[annealer], validation_data=(x_val, y_val))

