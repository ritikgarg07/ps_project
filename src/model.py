# import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, optimizers

# batch_size = 10
patch_size = 32

# def mrae(y_target, y_predicted):
#     m = tf.reduce_sum(tf.divide(tf.abs(y_target - y_predicted), y_target))
#     # TODO: Change hardcoded value
#     # pixel_count = 32*32*31
    # return tf.reduce_mean(m, axis = 0)

def unet(input_size = (patch_size, patch_size, 3)):

    rgb = layers.Input(input_size)
   
    # ENCODER layers 
    conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='valid')(rgb)  
    conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(conv1)
    conv3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid')(conv2)
    conv4 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid')(conv3)
    
    # Decoder layers
    # TODO: CHECK CONCATENATE AXIS
    upconv1 = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=1, activation='relu', padding='valid')(conv4)
    merge1 = layers.concatenate([upconv1, conv3], axis = -1)
    
    upconv2 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='valid')(merge1)
    merge2 = layers.concatenate((upconv2, conv2), axis = -1)
    
    upconv3 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='valid')(merge2)
    merge3 = layers.concatenate((upconv3, conv1), axis = -1)
    
    # reconstruction of rgb
    upconv4 = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, activation='relu', padding='valid')(merge3)
    
    # final hyperspectral layer
    # 1x1 convolution for now
    hyper = layers.Conv2D(filters=31, kernel_size=1, strides=1)(upconv4)


    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(epsilon= 1e-8, learning_rate=0.001), loss = 'mse', metrics = tf.keras.metrics.MeanAbsoluteError())

    # print(model.summary())
    return model

