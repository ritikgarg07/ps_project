import h5py
from PIL import Image
import random
import numpy as np 
import os
import glob
import yaml

# ! Specifiy absolute path here
with open('/workspaces/ps_project/config.yaml') as file:
    config = yaml.safe_load(file)



# Class encapsulating the data preparation functionality
class DataPrepare(object):

    # @param src_path: path to raw images
    # @param dest_path: path to store
    # @param size: patch_size of images  
    def __init__(self, src_path, dest_path, size):
        self.src_path = src_path
        self.dest_path = dest_path        
        self.size = size

    # @param id: id of image to generate dataset for
    # Stores the ip and op image in h5 file using self 
    def __crop_store(self, id):

        ip_file = glob.glob(self.current_path + '*.bmp')[0]
        op_files = sorted(glob.glob(self.current_path + '*.png'))
        
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
                
                name = str(id).zfill(2) + '_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                a = ip_image[start_j:  end_j, start_i: end_i]
                self.ip.create_dataset(name, data = a)
                start_j += 16
                end_j = start_j + 32


            start_i += 16
            end_i = start_i + 32

        b = np.zeros(shape = (img_size, img_size, config["wavelengths"]), dtype = np.uint16)
        for wv, op_file in enumerate(op_files):
            if id == 31:
                # ! OP Images for id = 31 are in RGBA, aplha channel is 255 throughout
                # Converting to 16 bit 
                op_image = np.array((Image.open(op_file).convert('I')))* (65535 / 255)
                op_image = op_image.astype(np.uint16)
                
            else:
                op_image = Image.open(op_file)
                op_image = np.array(Image.open(op_file), dtype = np.uint16)

            b[:,:,wv] = np.asarray(op_image)
        
        start_i = 0
        end_i = 32
        
        for i in range(0, pieces):
            start_j = 0
            end_j = 32
            for j in range(0, pieces):
                
                # template: b[j * self.size: (j + 1) * self.size, i * self.size: (i + 1) * self.size, :]
                name = str(id).zfill(2) + '_' + str(i).zfill(2) + '_' + str(j).zfill(2)
                a = b[start_j:  end_j, start_i: end_i, :]
                self.op.create_dataset(name, data = a)
                print(start_i, end_i, start_j, end_j)
                start_j += 16
                end_j = start_j + 32
 
            start_i += 16
            end_i = start_i + 32
                

    # Generates training set from images_range 
    def __create_train(self, start, end):
        self.file = h5py.File(self.dest_path + 'train2.h5', 'w')
        self.ip = self.file.create_group('ip')
        self.op = self.file.create_group('op')

        for id in self.image_range[start: end]:
            self.current_path = self.src_path + str(id) + '/'
            self.__crop_store(id)

    # Generate validation set from images_range
    def __create_validation(self, start, end):
        self.file = h5py.File(self.dest_path + 'validation2.h5', 'w')
        self.ip = self.file.create_group('ip')
        self.op = self.file.create_group('op')

        for id in self.image_range[start: end]:
            self.current_path = self.src_path + str(id) + '/'
            self.__crop_store(id)

    # Generate test set from images_range
    # ! [start: ]; till end of images_range
    def __create_test(self, start):
        self.file = h5py.File(self.dest_path + 'test2.h5', 'w')
        self.ip = self.file.create_group('ip')
        self.op = self.file.create_group('op')

        for id in self.image_range[start:]:
            print(id)
            self.current_path = self.src_path + str(id) + '/'
            self.__crop_store(id)

    # Splits the image_range into test, validation and test
    # @param: number of images in each test, validation, test
    def split(self, train, validation, test):  

        # 0 - 31 images
        self.image_range = list(range(0,train + validation + test))

        # Shuffle the images_range
        random.shuffle(self.image_range)

        self.__create_train(0, train)
        self.__create_validation(train, train + validation)
        self.__create_test(train + validation)


# TODO: Move to main.py
prepare = DataPrepare(dest_path = config["base_dir"] + 'data/', src_path = config["base_dir"] + 'data/raw/', size = config["patch_size"])

prepare.split(train = config["train_no"], validation = config["validation_no"], test = config["test_no"])