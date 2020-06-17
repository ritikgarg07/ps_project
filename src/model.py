# import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, optimizers


# Metric MRAE
class MRAE(tf.keras.metrics.Metric):

    def __init__(self, name = 'mrae_metric', **kwargs):
        super(MRAE, self).__init__(name = name, **kwargs)
        self.mrae = self.add_weight(name='mrae', initializer = 'zeros')
    
    def update_state(self, y_true, y_pred):
        zero_protected_true = tf.maximum(y_true, 1e-5*tf.ones_like(y_true))
        self.mrae.assign_add(tf.reduce_mean(tf.divide(tf.abs(y_pred - y_true), zero_protected_true)))

    def result(self):
        return self.mrae

    def reset_states(self):
        self.mrae.assign(0.)

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
    conv1 = layers.Conv2D(filters = 31, kernel_size = 3, strides= 1, padding='same', activation='relu')(prgb)

    resb1 = resnet_block(conv1, input_dim, 31, 31)
    resb2 = resnet_block(resb1, input_dim, 31, 31)
    resb3 = resnet_block(resb2, input_dim, 31, 31)
    resb4 = resnet_block(resb3, input_dim, 31, 31)
    resb5 = resnet_block(resb4, input_dim, 31, 31)
    resb6 = resnet_block(resb5, input_dim, 31, 31)
   
    hyper = part_conv(resb6, input_dim, 31)
    hyper = layers.Conv2D(filters = 31, kernel_size = 1, strides = 1, padding='same', activation='sigmoid')(hyper)

    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError(), MRAE()])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)
        
    # print(model.summary())
    return model


def unet(input_size, wavelengths, pretrained_weights = None):

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
    hyper = layers.Conv2D(filters=wavelengths, kernel_size=1, strides=1, activation='sigmoid')(upconv4)


    # TODO: Change metric to mean relative absolute error as described in VIDAR paper
    model = tf.keras.Model(inputs = rgb, outputs = hyper)
    model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError(), MRAE()])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    # print(model.summary())
    return model

