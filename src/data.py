import h5py
import matplotlib.pyplot as plt
import numpy as np 
import os
from PIL import Image
import tensorflow as tf


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
                ip_np = np.array(ip_group.get(ip)).astype(np.float32)
                ip_np /= 255

                op_np = np.array(op_group.get(op)).astype(np.float32)
                op_np /= 65535

                yield (ip_np, op_np)

# Class for dataset
class DataSet(object):
    
    def __init__(self, batch_size, mode, path, patch_size, wavelengths):
        self.batch_size = batch_size
        self.path = path
        self.mode = mode
        self.patch_size = patch_size
        self.wavelengths = wavelengths

    def get_batch(self):
        return next(iter(self.ds))

    # Main function
    def load_data(self):
        
        self.ds = tf.data.Dataset.from_generator(Generator(self.path + self.mode + '.h5'), output_types=(tf.float32, tf.float32), output_shapes=((self.patch_size, self.patch_size, 3), (self.patch_size, self.patch_size, self.wavelengths)), args = [])


        self.ds = self.ds.batch(self.batch_size)
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)

        return self.ds

# TODO: CONFIGURE CONFIG FILE
def convert_image(prediction):
    
    for wv in range(31):
        b = np.empty((512, 512))
        for index, arr in enumerate(prediction[:]):
            i = index % 16
            j = index // 16
            b[i*32: (i+1)*32, j*32: (j+1)*32] = np.transpose(arr[:,:,wv])

        b = np.transpose(b)
        b = b * 65535
        b = np.array(b, dtype = np.uint16)
        a = Image.fromarray(b).convert('I;16')
        a.save('/workspaces/ps_project/results/' + str(wv + 1).zfill(2) + '.png')

def plot_spectrum_by_wv(prediction):
    for wv in range(31):
        patch = np.random.randint(0, 256)
        a = tf.reshape(prediction[patch,:,:,wv], (32*32, 1))
        plt.plot(a * 65535, label = 'prediction')
        
        with h5py.File('/workspaces/ps_project/data/test.h5', 'r') as hf:
            op_group = hf['/op']
            for index, op in enumerate(op_group):
                if index == patch:
                    b = np.reshape(np.array(op_group.get(op))[:, :, wv], (32*32, 1))
                    plt.plot(b, label = 'truth')
                    plt.legend()
                    plt.title(f"Patch: {patch} Wavelength: {wv}")
                    plt.savefig(f"/workspaces/ps_project/plots/by_wv/test_{wv}_{patch}.png")
                    plt.clf()
                else:
                    pass

def plot_spectrum_by_pixel(prediction):
    for i in range(20):
        patch = np.random.randint(0, 256)
        x = np.random.randint(0, 32)
        y = np.random.randint(0,32)

        plt.plot(prediction[patch, x, y, :] * 65535, label = 'prediction')
        with h5py.File('/workspaces/ps_project/data/test.h5', 'r') as hf:
            op_group = hf['/op']
            for index, op in enumerate(op_group):
                if index == patch:
                    b = np.array(op_group.get(op))[x, y, :]
                    plt.plot(b, label = 'truth')
                    plt.xlabel('wavelength')
                    plt.legend()
                    plt.title(f"Patch: {patch} X: {x} Y: {y}")
                    plt.savefig(f"/workspaces/ps_project/plots/by_pixel/test_{i}_{x}_{y}.png")
                    plt.clf()
                else:
                    pass