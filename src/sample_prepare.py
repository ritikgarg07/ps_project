import h5py
from PIL import Image
import random
import numpy as np 
import os
import glob
import yaml

with open('/workspaces/ps_project/config.yaml') as file:
    config = yaml.safe_load(file)


# Class encapsulating the data preparation functionality
class SamplePrepare(object):

    # @param src_path: path to raw images
    # @param dest_path: path to store
    # @param size: patch_size of images  
    def __init__(self, src_path, dest_path, size):
        self.src_path = src_path
        self.dest_path = dest_path        
        self.size = size

    # Stores the ip and op image in h5 file using self 
    def __crop_store(self, op_present = False):
        for ip_file in glob.glob(self.current_path + '*.bmp'):
            file_name = (ip_file.split('/')[-1])[:-4]

            # input image
            ip_image = np.array(Image.open(ip_file))
            img_size = ip_image.shape[0]
            border = 16
            pieces = img_size // (self.size - border)
            pieces -= 1

            start_i = 0
            end_i = 32
            
            for i in range(0, pieces):
                start_j = 0
                end_j = 32
                for j in range(0, pieces):
                    
                    name = file_name + '_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                    a = ip_image[start_j:  end_j, start_i: end_i]
                    self.ip.create_dataset(name, data = a)
                    start_j += 16
                    end_j = start_j + 32


                start_i += 16
                end_i = start_i + 32

            b = np.zeros(shape = (img_size, img_size, config["wavelengths"]), dtype = np.uint16)
            
            start_i = 0
            end_i = 32
            
            for i in range(0, pieces):
                start_j = 0
                end_j = 32
                for j in range(0, pieces):
                    
                    name = file_name + '_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                    a = b[start_j:  end_j, start_i: end_i, :]
                    self.op.create_dataset(name, data = a)
                    start_j += 16
                    end_j = start_j + 32

                start_i += 16
                end_i = start_i + 32
                    

    def prepare_sample(self):
        print("Creating image dataset...")
        self.file = h5py.File(self.dest_path + 'sample.h5', 'w')
        self.ip = self.file.create_group('ip')
        self.op = self.file.create_group('op')

        print(f"Output path: {self.dest_path}")
        self.current_path = self.src_path + '/'
        self.__crop_store()
        