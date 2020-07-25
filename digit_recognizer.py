# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

# reading the data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

X_train_org = (train.iloc[:,1:].values).astype('float32') # all pixel values
X_train_org /= 255 # standardizing the image data
y_train_org = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')
X_test /= 255 # standardizing the image data

#Convert train datset to (num_images, img_rows, img_cols, no. of channels) format 
X_train_org = X_train_org.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# visualizing some digit images
for i in range(0, 3):
    plt.subplot(130 + (i+1))
    plt.imshow(X_train_org[i,:,:,0], cmap=plt.get_cmap('gray'))
    plt.title(y_train_org[i]);
    

# one hot encoding for output labels
y_train_org = to_categorical(y_train_org, num_classes = 10)

X_train, X_val, y_train, y_val = train_test_split(X_train_org, y_train_org,
                                                  test_size = 0.10, random_state = 42)

# making the convolutional neural network architecture
model= Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', 
                  activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', 
                  activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 
                  activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', 
                  activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


print("input shape ", model.input_shape)
print("output shape ", model.output_shape)

optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)


model.compile(optimizer = optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                                height_shift_range=0.08, zoom_range=0.08)

gen.fit(X_train)

# make the learning rate half if the validation accuracy does not improve over
# three consecutive epochs
learning_rate_reduction = ReduceLROnPlateau( monitor='val_acc', patience = 3, 
                                              verbose = 1, factor = 0.5, min_lr = 0.00001)

history = model.fit_generator(gen.flow(X_train, y_train, batch_size = 64), 
                              steps_per_epoch = X_train.shape[0] // 64, epochs=30, 
                              validation_data = (X_val, y_val), verbose = 1,
                              callbacks = [learning_rate_reduction])

# save the trained model 
model.save("trained_model")

history_dict = history.history

# making predictions
predictions = model.predict_classes(X_test, verbose=0)

# generating submission file for Kaggle
submission = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)),
                          "Label": predictions})
submission.to_csv("DR.csv", index = False, header = True)
