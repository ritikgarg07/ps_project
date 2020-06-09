import numpy as np 
import tensorflow as tf
import os

def get_id(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    id = parts[-1]
    id = tf.strings.split(id, '.')
    id = id[0]
    return id

def decode_input(img):
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def decode_output(id):
    # !HARDCODED VALUE: CHANGE
    output = []
    for i in range(32):
        img = tf.strings.join(id, '_', str(i), )
        img = tf.image.decode_image(img)
        # TODO: SCALE OUTPUT IMAGES BY COMMON FACTOR
        img = tf.image.convert_image_dtype(img) 
        output.append(img)
    output = tf.stack(output, axis = 2)
    output = tf.reshape(output, (32,32,31))

# def process_path(file_path):
    # 

list_ds = tf.data.Dataset.list_files(str('/workspaces/ps_project/data/test/*.bmp'))
for f in list_ds.take(5):
    # get_labels(f)
    # pass
    print(f)