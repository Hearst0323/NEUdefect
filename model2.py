from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications import MobileNetV2
import parameter

def MyMobileNetV2():
    
    input_shape = (parameter.IMAGE_SIZE_X, parameter.IMAGE_SIZE_Y, parameter.channel)
    
    model = MobileNetV2(input_shape = input_shape,
                        include_top = False,
                        weights = 'imagenet')
    
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(6, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    
    model = Model(inputs = model.input, outputs = predictions)
    
    optimizer = Adam(lr=0.001)
    
    loss = "categorical_crossentropy"
    
    for layer in model.layers[0:155]:
        layer.trainable = False
        
    for layer in model.layers[155:-1]:
        layer.trainable = True
    
    model.compile(optimizer=optimizer,    
                  loss=loss,
                  metrics=["accuracy"])

    model.summary()
    
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    
    
    
    return model
    
