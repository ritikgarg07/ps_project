import h5py
from PIL import Image
import random
import numpy as np 
import os
import glob

class DataPrepare(object):

    def __init__(self, src_path, dest_path, size):
        self.src_path = src_path
        self.dest_path = dest_path        
        self.size = size

    def __crop_store(self, id):
        ip_file = glob.glob(self.current_path + '*.bmp')[0]
        op_files = sorted(glob.glob(self.current_path + '*.png'))
        
        # input image
        ip_image = np.array(Image.open(ip_file))
        img_size = ip_image.shape[0]
        pieces = img_size // self.size
        for i in range(0, pieces):
            for j in range(0, pieces):
                name = str(id).zfill(2) + '_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                a = ip_image[i * self.size:  (i + 1) * self.size, j * self.size: (j + 1) * self.size]
                self.ip.create_dataset(name, data = a)
        

        for wv, op_file in enumerate(op_files):
            b = np.empty(shape = (img_size, img_size, 31), dtype = np.uint16)
            if id == 31:
                # ! OP Images for id = 31 are in RGBA, aplha channel is 255 throughout
                # Converting to 16 bit 
                op_image = np.array((Image.open(op_file).convert('I')))* (65535 /255)
                op_image = op_image.astype(np.uint16)
                
            else:
                op_image = np.array(Image.open(op_file), dtype = np.uint16)
            b[:,:,wv] = np.asarray(op_image)
       
        for i in range(0, pieces):
            for j in range(0, pieces):
                # template: b[j * self.size: (j + 1) * self.size, i * self.size: (i + 1) * self.size, :]
                name = str(id).zfill(2) + '_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                a = b[i * self.size:  (i + 1) * self.size, j * self.size: (j + 1) * self.size, :]
                self.op.create_dataset(name, data = a)
                
    def __create_train(self, start, end):
        self.file = h5py.File(self.dest_path + 'train.h5', 'w')
        self.ip = self.file.create_group('ip')
        self.op = self.file.create_group('op')

        for id in self.image_range[start: end]:
            self.current_path = self.src_path + str(id) + '/'
            self.__crop_store(id)

    def __create_validation(self, start, end):
        self.file = h5py.File(self.dest_path + 'validation.h5', 'w')
        self.ip = self.file.create_group('ip')
        self.op = self.file.create_group('op')

        for id in self.image_range[start: end]:
            self.current_path = self.src_path + str(id) + '/'
            self.__crop_store(id)

    def __create_test(self, start):
        self.file = h5py.File(self.dest_path + 'test.h5', 'w')
        self.ip = self.file.create_group('ip')
        self.op = self.file.create_group('op')

        for id in self.image_range[start:]:
            self.current_path = self.src_path + str(id) + '/'
            self.__crop_store(id)

    # number of images in each
    def split(self, train, validation, test):  

        # 0 - 31 images
        self.image_range = list(range(0,32))
        random.shuffle(self.image_range)

        self.__create_train(0, train)
        self.__create_validation(train, train + validation)
        self.__create_test(train + validation)


prepare = DataPrepare(dest_path = '/workspaces/ps_project/data/', src_path = '/workspaces/ps_project/data/raw/', size = 32)
prepare.split(29, 2, 1)