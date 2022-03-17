import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import adam_v2
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D,BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

from pca import digits_pca

digits = pd.read_csv('data/train.csv')

y = digits["label"]
x = digits.drop(labels = ["label"],axis = 1)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)
x_train = x_train/255.0
x_val = x_val/255.0
x_train = x_train.values.reshape(-1,28,28,1)
x_val = x_val.values.reshape(-1,28,28,1)
#x_train_final = x_train_reshape.astype("float32")/255

y_train = to_categorical(y_train, num_classes = 10)
y_val = to_categorical(y_val, num_classes = 10)
print("Train data size :" + str(len(x_train)))
print("Validation data size :" + str(len(x_val)))

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

model.summary()
datagen = ImageDataGenerator(zoom_range=0.1,height_shift_range=0.1,width_shift_range=0.1,rotation_range=10)
#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer= adam_v2.Adam(learning_rate=1e-4)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# set learning rate
# annealer = LearningRateScheduler (lambda x:1e-3*0.9**x)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
# fit model
epochs = 10 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

history = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_val,y_val),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])