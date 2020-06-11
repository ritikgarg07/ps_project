import numpy as np 
import tensorflow as tf
import tensorboard
import os
import time
import h5py

AUTOTUNE = tf.data.experimental.AUTOTUNE

class Generator(object):
    def __init__(self, file):
        self.file = file
        
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            ip_group = hf['/ip']
            op_group = hf['/op']

            # ! Normalises ip (RGB) images by 255 {2^8 - 1} and op (Hyper) by 65536 {2^16 - 1}
            # ! Returns float32, should match with dataset.load_data()
            # TODO: float32/ float64?
            for ip, op in zip(ip_group, op_group):
                ip_np = np.array(ip.group_get(ip)).astype(np.float32)
                ip_np /= 255

                op_np = np.array(op_group.get(op)).astype(np.float32)
                op_np /= 65535

                yield (ip_np, op_np)

# Class for dataset
class DataSet(object):
    
    def __init__(self, batch_size, path = '/workspaces/ps_project/data/'):
        self.batch_size = batch_size
        self.path = path
        
    # Main function
    def load_data(self, shuffle_buffer_size = 1000, train = False, test = True, validation = False):
        
        # !TODO: Appropriate way to pass 32 = patch_size here
        # ? Config file
        self.ds = tf.data.Dataset.from_generator(Generator(self.path + 'test/'), output_types=(tf.float32, tf.float32), output_shapes=((32, 32, 3), (32, 32, 31)), args = [])

        self.ds = self.ds.cache('/workspaces/ps_project/cache/')
        self.ds = self.ds.batch(self.batch_size)
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)
        
        return self.ds
    
    def get_batch(self):
        return next(iter(self.ds))

dataset = DataSet(batch_size=5)
dataset.load_data()