import numpy as np 
import tensorflow as tf
import os
AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataSet(object):
    def __init__(self, batch_size, path = '/workspaces/ps_project/data/'):
        self.batch_size = batch_size
        # self.dataset = dataset
        # self.train = train
        # self.validation = validation
        # self.test = test
        self.path = path
    
    # Takes the file_path to a bmp(RGB) input and outputs the image name
    def __get_id(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        id = parts[-1]
        id = tf.strings.split(id, '.')
        id = id[0]
        return id
    
    # Takes the file_path to a bmp(RGB) input and outputs the image as a tensor of shape (im_size, im_size, 3)
    # ! Normalises uses implicit normalisation from uint8 to float 32 ie- [0,255] to [0,1]
    def __decode_input(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    # Takes the file_path to a png(Hyperpectral) output image and outputs the image as a tensor of shape (im_size, im_size, 1)
    # ! No normalisation done for now, returns tensor of dtype uint8
    # TODO: Implement normalisation
    # TODO: Lookup channels for png image
    def __decode_output(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img)
        img = tf.image.convert_image_dtype(img, dtype=tf.uint8)
        return img 

    # Wrapper function
    # Takes file_path to a bmp(RGB) input image and returns a tuple of input image, output image (stacked)
    # (tensor of shape (32, 32, 3), tensor of shape (32, 32, 31))
    def __process_ds(self, file_path):
        
        ip = self.__decode_input(file_path)    # input image tensor

        # Get input image name
        id_image = self.__get_id(file_path)
        id_string = str(id_image)
        id_string = id_string[id_string.find("'")+1:]
        id_string = id_string[: id_string.find("'")]
        id_string += '_'
        
        # List of output image file_paths for the input image
        list_op = tf.io.gfile.glob(self.path + id_string + '*.png')
        
        # A list of output image tensors for input images
        # ! uint8 should match with type in decode_output
        images_op = [tf.zeros((32,32,1), dtype = tf.uint8) for i in range(31)]
        for index, file_path in enumerate(list_op):
            images_op[index] = self.__decode_output(file_path)
        
        # Generate single tensor of shape (32, 32, 31) from 31 tensors of shape (32, 32)
        op = tf.stack(images_op, axis = 2)
        op = tf.reshape(op, (32,32,31))

        return (ip, op)

    def load_process(self, shuffle_buffer_size = 1000, train = True, test = False, validation = False):
        # TODO: Implement test and validation 

        if train:
            self.path = self.path+ 'train/'

        # List of input images in given folder defined by path variable
        self.list_ds = tf.data.Dataset.list_files(str(self.path + '*.bmp'))

        # List of tuples (input, output) in given folder defined by path variable
        self.labelled_ds = self.list_ds.map(self.__process_ds, num_parallel_calls=AUTOTUNE)
        # labelled_ds = self.labelled_ds.cache()

        # self.labelled_ds = self.labelled_ds.shuffle(buffer_size = shuffle_buffer_size)
        self.labelled_ds = self.labelled_ds.repeat()
        self.labelled_ds = self.labelled_ds.batch(self.batch_size)

        self.labelled_ds = self.labelled_ds.prefetch(buffer_size=AUTOTUNE)

    def get_batch(self):
        return next(iter(self.labelled_ds))


dataset = DataSet(batch_size=10)
dataset.load_process()
# while True:
#     ip, op = dataset.get_batch()
#     print(ip.shape)
#     print(op.shape)
# mport time
# default_timeit_steps = 1000

import time
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
  start = time.time()
#   it = iter(ds)
  for i in range(steps):
    batch = ds.get_batch()
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(10*steps/duration))

  timeit(dataset)