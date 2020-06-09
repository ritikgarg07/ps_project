import numpy as np 
import tensorflow as tf
import os

path = '/workspaces/ps_project/data/test/'

def get_id(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    id = parts[-1]
    id = tf.strings.split(id, '.')
    id = id[0]
    return id

def decode_input(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def decode_output(file_path):
    # !HARDCODED VALUE: CHANGE
    img = tf.io.read_file(file_path)
    # TODO: CHANNELS?
    img = tf.image.decode_image(img)
    # TODO: SCALE OUTPUT IMAGES BY COMMON FACTOR
    return img 

def process_ds(file_path):
    
    ip = decode_input(file_path)
    id_ = get_id(file_path)
    id_string = str(id_)
    id_string = id_string[id_string.find("'")+1:]
    id_string = id_string[: id_string.find("'")]
    id_string += '_'
    
    list_op = tf.io.gfile.glob(path + id_string + '*.png')
    # print(len(list_op))
    if(len(list_op) != 31):
        print(id_string)
    list_op = tf.convert_to_tensor(list_op)
    images_op = tf.map_fn(decode_output, list_op, dtype = tf.uint8)       
    op = tf.stack(images_op, axis = 2)
    op = tf.reshape(op, (32,32,31))
    return (ip, op)


list_ds = tf.data.Dataset.list_files(str(path + '*.bmp'))
for f in list_ds.take(256):
    a = process_ds(f)
#     # print(a[1].shape)
# labelled_ds = list_ds.map(process_ds)
