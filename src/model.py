# import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, optimizers

batch_size = 10
patch_size = 32

def unet(input_size = (patch_size, patch_size, 3)):

    rgb = layers.Input(input_size)
   
    # ENCODER layers 
    conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='valid')(rgb)  
    conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(conv1)
    conv3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid')(conv2)
    conv4 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid')(conv3)
    
    # Decoder layers
    upconv1 = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid')(conv4)
    merge1 = layers.concatenate([upconv1, conv3], axis = 3)
    
    upconv2 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(merge1)
    merge2 = layers.concatenate((upconv2, conv2), axis=3)
    
    upconv3 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='valid')(merge2)
    merge3 = layers.concatenate((upconv3, conv1), axis=3)
    
    # output layer, filters = number of bands
    hyper = layers.Conv2DTranspose(filters=31, kernel_size=3, strides=1, activation='relu', padding='valid')(merge3)
    
    # TODO: Change loss and metric to mean relative absolute error as described in VIDAR paper
    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(learning_rate = 0.0001), loss = 'mse', metrics = 'mse')

    # print(model.summary())
    return model
