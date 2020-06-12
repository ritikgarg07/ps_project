# import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, optimizers

# batch_size = 10
patch_size = 32

def unet(input_size = (patch_size, patch_size, 3), pretrained_weights = None):

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
    merge2 = layers.concatenate((upconv2, conv2), axis = 3)
    
    upconv3 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='valid')(merge2)
    merge3 = layers.concatenate((upconv3, conv1), axis = 3)
    
    # reconstruction of rgb
    upconv4 = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, activation='relu', padding='valid')(merge3)
    
    # final hyperspectral layer
    # 1x1 convolution for now
    hyper = layers.Conv2D(filters=31, kernel_size=1, strides=1, activation='sigmoid')(upconv4)


    # TODO: Change loss and metric to mean relative absolute error as described in VIDAR paper
    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), loss = 'mse', metrics = tf.keras.metrics.MeanAbsoluteError())
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    # print(model.summary())
    return model

