import numpy as np 
import tensorflow as tf
import os
# import time
AUTOTUNE = tf.data.experimental.AUTOTUNE

# TODO: PATH WOULD BE A VARIABLE PASSED ON FROM MAIN.PY
path = '/workspaces/ps_project/data/train/'

# Takes the file_path to a bmp(RGB) input and outputs the image name
def get_id(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    id = parts[-1]
    id = tf.strings.split(id, '.')
    id = id[0]
    return id

# Takes the file_path to a bmp(RGB) input and outputs the image as a tensor of shape (im_size, im_size, 3)
# ! Normalises uses implicit normalisation from uint8 to float 32 ie- [0,255] to [0,1]
def decode_input(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

# Takes the file_path to a png(Hyperpectral) output image and outputs the image as a tensor of shape (im_size, im_size, 1)
# ! No normalisation done for now, returns tensor of dtype uint8
# TODO: Implement normalisation
# TODO: Lookup channels for png image
def decode_output(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img)
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
    return img 

# Wrapper function
# Takes file_path to a bmp(RGB) input image and returns a tuple of input image, output image (stacked)
# (tensor of shape (32, 32, 3), tensor of shape (32, 32, 31))
def process_ds(file_path):
    
    ip = decode_input(file_path)    # input image tensor

    # Get input image name
    id_ = get_id(file_path)
    id_string = str(id_)
    id_string = id_string[id_string.find("'")+1:]
    id_string = id_string[: id_string.find("'")]
    id_string += '_'
    
    # List of output image file_paths for the input image
    list_op = tf.io.gfile.glob(path + id_string + '*.png')
    
    # A list of output image tensors for input images
    # ! uint8 should match with type in decode_output
    images_op = [tf.zeros((32,32,1), dtype = tf.uint8) for i in range(31)]
    for index, file_path in enumerate(list_op):
        images_op[index] = decode_output(file_path)
    
    # Generate single tensor of shape (32, 32, 31) from 31 tensors of shape (32, 32)
    op = tf.stack(images_op, axis = 2)
    op = tf.reshape(op, (32,32,31))
    
    return (ip, op)
    
# List of input images in given folder defined by path variable
list_ds = tf.data.Dataset.list_files(str(path + '*.bmp'))

# List of tuples (input, output) in given folder defined by path variable
labelled_ds = list_ds.map(process_ds, num_parallel_calls=AUTOTUNE)

# TODO: IMPLEMENT LOAD/ PREPARE TRAINING/ GENERATOR FUNCTION
# List of input images in given folder defined by path variable
list_ds = tf.data.Dataset.list_files(str(path + '*.bmp'))

# List of tuples (input, output) in given folder defined by path variable
labelled_ds = list_ds.map(process_ds, num_parallel_calls=AUTOTUNE)

def load_process(labelled_ds, shuffle_buffer_size = 1000, train = True, test = False, validation = False):
    # TODO: Implement test and validation 
    
    # labelled_ds = labelled_ds.shuffle(buffer_size = shuffle_buffer_size)
    labelled_ds = labelled_ds.repeat()
    labelled_ds = labelled_ds.batch(10)

    labelled_ds = labelled_ds.prefetch(buffer_size=AUTOTUNE)
    return labelled_ds

def get_batch(labelled_ds):
    return next(iter(labelled_ds))

labelled_ds = load_process(labelled_ds)
# image, label = get_batch(labelled_ds)
