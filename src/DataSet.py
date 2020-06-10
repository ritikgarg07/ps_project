import numpy as np 
import tensorflow as tf
import tensorboard
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Class for dataset
class DataSet(object):
    def __init__(self, batch_size, path = '/workspaces/ps_project/data/'):
        self.batch_size = batch_size
        self.path = path
    
    # Takes the file_path to a bmp(RGB) input and outputs the image name
    def __get_id(self, file_path):
        parts = file_path.split('/')
        id = parts[-1]
        id = id.split('.')
        id = id[0]
        return id    
    
    # Takes the file_path to a bmp(RGB) input and returns the image as a tensor of shape (32, 32, 3)
    # ! Normalises using implicit normalisation from uint8 to float32 (to [0,1])
    def __decode_input(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype = tf.dtypes.float32)
        img = tf.reshape(img, (32, 32, 3))
        return img

    # Takes file_path to a png(Hyper) output image and returns the image as a tensor of shape (32, 32, 1)
    # ! Normalises using implicit normalisation from uint16 to float32 (to [0,1])
    def __decode_output(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1, dtype = tf.dtypes.uint16)
        img = tf.image.convert_image_dtype(img, dtype = tf.dtypes.float32)
        return img 

    # Takes the list of file_paths corr. to a input image and outputs the corr. output stacked image as a tensor of shape (32, 32, 31)
    def __parse_output(self, path_list):
        tensor_list = [self.__decode_output(file_path) for file_path in path_list]
        op = tf.stack(tensor_list, axis = 2)
        op = tf.reshape(op, (32, 32, 31))
        return op
        
    # Main function
    def load_process(self, shuffle_buffer_size = 1000, train = False, test = False, validation = False):
        
        if (not train) & (not test) & (not validation):
            print('Error! Please select one')
            return -1
        if train:
            folder = 'train/'
        if validation:
            folder = 'validation/'
        if test:
            folder = 'test/'

        # List of paths to input images in given folder defined by path variable
        self.list_ip = tf.io.gfile.glob(str(self.path + folder + '*.bmp'))

        # List of 'ids' of input images ie- 'sampleno_rowno_column_no'
        self.list_id = list(map(self.__get_id, self.list_ip))

        # List of lists of paths to output images for each input image
        # Structure: [ [op for ip1], [op for ip2], ...]
        self.list_op = list(map(lambda f: tf.io.gfile.glob(self.path + folder + f + '*.png'), self.list_id))
        
        # List of op stacked images for each ip
        # Structure: [ tensor(32, 32, 31) for op1, tensor(32, 32, 31 for op2), ...]
        # Decodes op images from paths to tensors
        self.tensor_list = [self.__parse_output(f) for f in self.list_op[:2]]
    
        # Temp dataset of only ip paths
        self.ds1 = tf.data.Dataset.from_tensor_slices(self.list_ip)
        self.ds1 = self.ds1.cache()
        # Decoding ip images from paths to tensors
        self.ds1 = self.ds1.map(self.__decode_input)

        # Temp dataset of only op tensors (has been parsed from images to tensors at tensor_list)
        self.ds2 = tf.data.Dataset.from_tensor_slices(self.tensor_list)
        
        # Final dataset: tuple of ip_tensor, op_tensor
        # ie- (tensor of shape (32, 32, 3), tensor of shape(32, 32, 31))
        self.ds = tf.data.Dataset.zip((self.ds1, self.ds2))

        # labelled_ds = self.labelled_ds.cache()
        # self.labelled_ds = self.labelled_ds.shuffle(buffer_size = shuffle_buffer_size)
        self.ds = self.ds.batch(self.batch_size)
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)

        return self.ds
    
    def get_batch(self):
        return next(iter(self.ds))

