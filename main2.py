import numpy as np
import pandas as pd
import cv2

import model
import model2

df = pd.read_csv('C:/Users/20015/jobs/20200525/info2.csv')

train_id = df['train']
train_ladbel = df['train_label']
val_id = df['val']
val_label = df['val_label']

x_data_train = np.zeros([1440, 200, 200, 3])
y_data_train = np.zeros(1440)

x_data_val = np.zeros([360, 200, 200, 3])
y_data_val = np.zeros(360)

for i in range(0, 1440):
    
    img = cv2.imread('C:/Users/20015/jobs/20200525/NEU surface defect database/' + train_id[i])   
    x_data_train[i, :, :, :] = img[:, :, :]
    y_data_train[i] = train_ladbel[i]
    
for i in range(0, 360):
    
    img = cv2.imread('C:/Users/20015/jobs/20200525/NEU surface defect database/' + val_id[i])
    x_data_val[i, :, :, :] = img[:, :, :]
    y_data_val[i] =  val_label[i]
    
from keras.utils import np_utils

x_data_train_normal = x_data_train / 255
y_data_train_onehot = np_utils.to_categorical(y_data_train)

x_data_val_normal = x_data_val / 255
y_data_val_onehot = np_utils.to_categorical(y_data_val)

from keras.applications import MobileNetV2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

modelpath = 'C:/Users/20015/.spyder-py3/surfacedefect/bestmodel1.h5'
checkpoint = ModelCheckpoint(modelpath, monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
callbacks_list = [checkpoint, reduce_lr]

#model = MobileNetV2(input_shape = (200, 200, 1), 
#                    include_top = True, 
#                    weights = None, 
#                    classes = 6)

#for layer in model.layers[:-1]:
#    layer.trainable = True
    
    
#model.summary()

model = model2.MyMobileNetV2()

#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zca_whitening=False,
                             rotation_range=200,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest') 

#datagen.fit(x_data_train_normal)

train_history = model.fit(x_data_train_normal, 
                          y_data_train_onehot, 
                          batch_size = 32,
                          epochs = 10,
                          verbose = 1,
                          validation_data = (x_data_val_normal, y_data_val_onehot)
                          )

import matplotlib.pyplot as plt

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.show()
    
def show_loss_history(train_history, loss, val_loss):
    plt.plot(train_history.history[loss])
    plt.plot(train_history.history[val_loss])
    plt.title("Loss History")
    plt.ylabel(loss)
    plt.xlabel('Epoch')
    plt.show()
    
    
show_train_history(train_history, "accuracy", "val_accuracy")
show_train_history(train_history, "loss", "val_loss")

b = model.predict(x_data_train_normal[:,:,:,:])
