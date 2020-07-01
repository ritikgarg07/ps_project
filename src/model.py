# import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, optimizers


# Metric MRAE
def mrae(y_true, y_pred):
    y_t = tf.math.maximum(y_true, 0.001*tf.ones_like(y_true))
    return tf.reduce_mean(tf.divide(tf.abs(y_true - y_pred), y_t), axis = -1)

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred), axis = -1)

# normalises input by 1p1/1p0 as described in partial conv padding paper
def part_conv(input, input_dim, input_filters):    
    p0 = tf.ones((1, input_dim, input_dim, 1))
    ratio = layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'same', use_bias = False, kernel_initializer=tf.constant_initializer(1.0), trainable = False)(p0)
    ratio2 = tf.broadcast_to(ratio, (1, input_dim, input_dim, input_filters))
    ratio3 = tf.math.divide(9*tf.ones((1, input_dim, input_dim, input_filters)), ratio)
    part_conv = layers.Multiply()([input, ratio3])
    return part_conv

# resnet block
def resnet_block(input, input_dim, input_filters, output_filters):
    x = part_conv(input, input_dim, input_filters)
    x = layers.Conv2D(output_filters, 3, 1, padding='same', activation=None)(x)
    x = part_conv(x, input_dim, output_filters)
    x = layers.Conv2D(output_filters, 3, 1, padding='same', activation='relu')(x)
    x = layers.Add()([x, input])
    return x

def resnet(input_dim, wavelengths = 31, pretrained_weights = None):
    
    rgb = layers.Input((input_dim, input_dim, 3))
    prgb = part_conv(rgb, input_dim, input_filters = 3)
    conv1 = layers.Conv2D(filters = 32, kernel_size = 3, strides= 1, padding='same', activation='relu')(prgb)

    resb1 = resnet_block(conv1, input_dim, 32, 32)
    resb2 = resnet_block(resb1, input_dim, 32, 32)
    resb3 = resnet_block(resb2, input_dim, 32, 32)
    resb3 = layers.Conv2D(filters = 64, kernel_size=3, strides=1, padding='same', activation='relu')(resb3)
    resb4 = resnet_block(resb3, input_dim, 64, 64)
    resb5 = resnet_block(resb4, input_dim, 64, 64)
    resb6 = resnet_block(resb5, input_dim, 64, 64)
    hyper = part_conv(resb6, input_dim, 64)
    hyper = layers.Conv2D(filters = 31, kernel_size = 1, strides = 1, padding='same', activation='sigmoid')(hyper)

    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), loss = 'mse', metrics = [mrae, mse])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        
    # print(model.summary())
    return model

def resnet2(input_dim, wavelengths = 31, pretrained_weights = None):
    
    rgb = layers.Input((input_dim, input_dim, 3))
    prgb = part_conv(rgb, input_dim, input_filters = 3)
    conv1 = layers.Conv2D(filters = 32, kernel_size = 3, strides= 1, padding='same', activation='relu')(prgb)

    resb1 = resnet_block(conv1, input_dim, 32, 32)
    resb2 = resnet_block(resb1, input_dim, 32, 32)
    resb2 = part_conv(resb2, input_dim, input_filters = 32)
    resb3 = layers.Conv2D(filters = 64, kernel_size=3, strides=1, padding='same', activation='relu')(resb2)
    resb4 = resnet_block(resb3, input_dim, 64, 64)
    resb5 = resnet_block(resb4, input_dim, 64, 64)
    resb6 = part_conv(resb5, input_dim, input_filters = 64)
    resb6 = layers.Conv2D(filters = 128, kernel_size = 3, strides=1, padding='same', activation='relu')(resb6)
    resb7 = resnet_block(resb6, input_dim, 128, 128)
    resb8 = resnet_block(resb6, input_dim, 128, 128)
    
    hyper = part_conv(resb6, input_dim, 128)
    hyper = layers.Conv2D(filters = 31, kernel_size = 1, strides = 1, padding='same', activation='sigmoid')(hyper)

    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), loss = 'mse', metrics = [mrae, mse])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        
    # print(model.summary())
    return model


def unet(input_size, wavelengths, pretrained_weights = None):

    rgb = layers.Input(input_size)
    input_dim = 32
    prgb = part_conv(rgb, input_dim, input_filters = 3)
    
    # ENCODER layers 
    conv1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(prgb)  
    conv1 = part_conv(conv1, input_dim, input_filters=32)
    conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(conv1)
    conv2 = part_conv(conv2, input_dim, input_filters=64)
    conv3 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')(conv2)
    conv3 = part_conv(conv3, input_dim, input_filters=128)
    conv4 = layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')(conv3)
    
    # Decoder layers
    upconv1 = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=1, activation='relu', padding='same')(conv4)
    merge1 = layers.concatenate([upconv1, conv3], axis = 3)
    
    upconv2 = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, activation='relu', padding='same')(merge1)
    merge2 = layers.concatenate((upconv2, conv2), axis = 3)
    
    upconv3 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, activation='relu', padding='same')(merge2)
    merge3 = layers.concatenate((upconv3, conv1), axis = 3)
    
    # reconstruction of rgb
    upconv4 = layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, activation='relu', padding='same')(merge3)
    
    # final hyperspectral layer
    # 1x1 convolution for now
    # upconv4 = part_conv(upconv4, input_dim, input_filters=3)
    hyper = layers.Conv2D(filters=wavelengths, kernel_size=1, strides=1, activation='sigmoid', padding = 'same')(upconv4)


    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), loss = 'mse', metrics = [mrae, mse])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

