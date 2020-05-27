import pandas as pd
import numpy as np
import cv2

import parameters
import model

df = pd.read_csv('C:/Users/20015/jobs/20200511/train.csv')

load_num = 5600

label = df['label']
imageid = df['image_id']

x_train = np.zeros([load_num, parameters.IMAGE_SIZE_Y, parameters.IMAGE_SIZE_X, parameters.channel])
y_train = np.zeros(load_num)

import random

ran = 0
items = []

for ran in range(0, load_num):
    items.append(ran)


for i in random.sample(items, load_num):
    
    img = cv2.imread('C:/Users/20015/jobs/20200511/C1-P1_Train/resize/' + imageid[i])
    print(imageid[i])
    
    if(label[i] == 'A'):
        y_train[i] = 0
    elif(label[i] == "B"):
        y_train[i] = 1
    else:
        y_train[i] = 2
    
    for x in range(0, parameters.IMAGE_SIZE_X):
        for y in range(0, parameters.IMAGE_SIZE_Y):
            for c in range(0, parameters.channel):
                x_train[i, y, x, 2 - c] = img[y, x, c]
    
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zca_whitening=False,
                             rotation_range=200,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest') 

split_index = 0.2

y_train_onehot = np_utils.to_categorical(y_train)  
x_train_normal = x_train / 255

y_train_onehot_val = y_train_onehot[round(load_num * (1 - split_index)):load_num]
x_train_val = x_train_normal[round(load_num * (1 - split_index)):load_num]

y_train_onehot_val = y_train_onehot_val.astype('float16')
x_train_val = x_train_val.astype('float16')


y_train_onehot_train = y_train_onehot[0:round(load_num * (1 - split_index))]
x_train_train = x_train_normal[0:round(load_num * (1 - split_index))]

y_train_onehot_train = y_train_onehot_train.astype('float16')
x_train_train = x_train_train.astype('float16')

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
import h5py

mymodel = model.MangoModel()   

modelpath = 'C:/Users/20015/.spyder-py3/Mango/bestmodel5.h5'
checkpoint = ModelCheckpoint(modelpath, monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7)
callbacks_list = [checkpoint, reduce_lr]

mymodel.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
mymodel.summary()

datagen.fit(x_train_train)
                                      
train_history = mymodel.fit(datagen.flow(x_train_train, y_train_onehot_train, batch_size = 32),
                            epochs = 200,
                            callbacks = callbacks_list, 
                            validation_data = (x_train_val, y_train_onehot_val),
                            verbose = 1)

#train_history = mymodel.fit(x = x_train_train, 
#                            y = y_train_onehot_train, 
#                            batch_size = 32,
#                            epochs = 50,
#                            callbacks = callbacks_list, 
#                            validation_data = (x_train_val, y_train_onehot_val),
#                            verbose = 1)

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


from keras.models import load_model

model = load_model('C:/Users/20015/.spyder-py3/Mango/bestmodel4.h5')

a = np.argmax(model.predict(x_train_val), axis = 1)

for i in range(round(len(label) * (1 - split_index)), len(imageid)):
    img = cv2.imread('C:/Users/20015/jobs/20200511/C1-P1_Train/resize/' + imageid[i])
    cv2.putText(img, 'label = ' + str(np.argmax(y_train_onehot_val[i - 4480], axis = 0)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'predict = ' + str(a[i - 4480]), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite('C:/Users/20015/jobs/20200511/C1-P1_Train/result/' + str(i) + '.jpg', img)
