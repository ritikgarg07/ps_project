import h5py
import matplotlib.pyplot as plt
import numpy as np 
import os
from PIL import Image
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Generator for dataset
class Generator(object):
    def __init__(self, file):
        self.file = file
        
    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            ip_group = hf['/ip']
            op_group = hf['/op']

            # ! Normalises ip (RGB) images by 255 {2^8 - 1} and op (Hyper) by 65535 {2^16 - 1}
            # ! Returns float32, should match with dataset.load_data()
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
        
        self.ds = tf.data.Dataset.from_generator(Generator(self.path + self.mode + '2.h5'), output_types=(tf.float32, tf.float32), output_shapes=((self.patch_size, self.patch_size, 3), (self.patch_size, self.patch_size, self.wavelengths)), args = [])

        self.ds = self.ds.batch(self.batch_size)
        self.ds = self.ds.prefetch(buffer_size=AUTOTUNE)

        return self.ds

# Takes in a prediction, aka training_examples x 32 x 32 x 31
# and generates 31 512 x 512 images, one for each wavelength
# ! NOTE: configured to handle only 1 image dataset
def convert_image(prediction, wavelengths, patch_size, image_size):
    
    for wv in range(wavelengths):
    
        # initialise empty placeholder array
        b = np.empty((image_size, image_size))

        for index, arr in enumerate(prediction[:]):
            # locate patch-size 
            i = index % 31
            j = index // 31

            # start_ and end_ are the locations of the starting and ending pixel that the patch refers to in the 512x512 image
            # refer to data_prepare.h5 for specifics of dataset preparation
            start_i = i*16 + 8 - (i==0)*8
            end_i = start_i + 16 + (i==0)*8 + (i==30)*8
           
            start_j = j*16 + 8 - (j==0)*8
            end_j = start_j + 16 + (j==0)*8 + (j==30)*8

            # take_s and and take_j are the starting and ending locations of the 512x512 image that the patch will provide information about
            take_si = 8 - (i==0)*8
            take_ei = take_si + 16 + (i==0)*8 + (i==30)*8 

            take_sj = 8 - (j==0)*8
            take_ej = take_sj + 16 + (j==0)*8 + (j==30)*8

            b[start_i: end_i, start_j: end_j] = arr[take_si:take_ei,take_sj:take_ej,wv]

        # rescale back for 16-bit png
        b = b * 65535
        b = np.array(b, dtype = np.uint16)
        a = Image.fromarray(b).convert('I;16')
        a.save('/workspaces/ps_project/results/' + str(wv + 1).zfill(2) + '.png')

def plot_spectrum_by_wv(prediction, wavelengths, patch_size, image_size, patch):
    for wv in range(31):
        patch = patch
        a = tf.reshape(prediction[patch,:,:,wv], (pow(patch_size, 2), 1))
        a = a[:512]
        plt.plot(a, label = 'prediction', color = 'orange')
        
        with h5py.File('/workspaces/ps_project/data/test.h5', 'r') as hf:
            op_group = hf['/op']
            ip_group = hf['/ip']

            for index, (op, ip) in enumerate(zip(op_group, ip_group)):

                if index == patch:
                    if wv == 25:
                        # red
                        red = np.reshape(np.array(ip_group.get(ip), dtype = np.float32)[:, :, 0], (pow(patch_size, 2), 1))
                        red /= 255
                        red = red[:512]
                        plt.plot(red, label = 'red', color = 'red')
                    if wv == 13:
                        green = np.reshape(np.array(ip_group.get(ip), dtype = np.float32)[:, :, 1], (pow(patch_size, 2), 1)) 
                        green /= 255
                        green = green[:512]
                        plt.plot(green, label = 'green', color = 'green')
                    if wv == 7:
                        blue = np.reshape(np.array(ip_group.get(ip), dtype = np.float32)[:, :, 2], (pow(patch_size, 2), 1)) 
                        blue /= 255
                        blue = blue[:512]
                        plt.plot(blue, label = 'blue', color = 'blue')
                    b = np.reshape(np.array(op_group.get(op), dtype = np.float32)[:, :, wv], (pow(patch_size, 2), 1))
                    b /= 65535
                    b = b[:512]
                    plt.plot(b, label = 'truth', color = 'black')
                    plt.legend()
                    plt.title(f"Patch: {patch} Wavelength: {wv}")
                    plt.savefig(f"/workspaces/ps_project/plots/by_wv/test_{wv}_{patch}.png")
                    plt.clf()
                else:
                    pass

def plot_spectrum_by_pixel(prediction, patch_size, image_size, patch, x_s, y_s):
    for i in range(32):
        patch = patch
        x = x_s + i
        y = y_s + i

        plt.plot(prediction[patch, x, y, :], label = 'prediction', color = 'orange')
        with h5py.File('/workspaces/ps_project/data/test.h5', 'r') as hf:
            op_group = hf['/op']
            ip_group = hf['/ip']
            for index, (op, ip) in enumerate(zip(op_group, ip_group)):
                if index == patch:
                    red = np.array(ip_group.get(ip))[x, y, 0] / 255
                    green = np.array(ip_group.get(ip))[x, y, 1] / 255
                    blue = np.array(ip_group.get(ip))[x, y, 2] / 255

                    plt.scatter(25, red, label = 'r', color = 'red')
                    plt.scatter(13, green, label = 'g', color = 'green')
                    plt.scatter(7, blue, label = 'b', color = 'blue')
                    b = np.array(op_group.get(op), dtype = np.float32)[x, y, :]
                    b /= 65535
                    plt.plot(b, label = 'truth', color = 'black')
                    plt.xlabel('wavelength')
                    plt.legend()
                    plt.title(f"Patch: {patch} X: {x} Y: {y}")
                    plt.savefig(f"/workspaces/ps_project/plots/by_pixel/test_{i}_{x}_{y}.png")
                    plt.clf()
                else:
                    pass

# creates the mrae plots
def save_results(unet, resnet, patch_size = 32):
    hf = h5py.File('/workspaces/ps_project/data/test2.h5', 'r')
    ip_group = hf['/ip']
    op_group = hf['/op']
    
    rgb = np.empty((512, 512, 3), dtype = np.float32)
    truth = np.empty((512, 512, 31), dtype = np.float32)
    u_pred = np.empty((512, 512, 31), dtype = np.float32)
    r_pred = np.empty((512, 512, 31), dtype = np.float32)
    u_mrae = np.empty((512, 512, 31), dtype = np.float32)
    r_mrae = np.empty((512, 512, 32), dtype = np.float32)

    for wv in range(31):
        for index, (ip, op) in enumerate(zip(ip_group, op_group)):
            ip_arr = np.array(ip_group.get(ip), dtype = np.float32) / 255
            op_arr = np.array(op_group.get(op), dtype = np.float32) / 65535
            i = index % 31
            j = index // 31

            start_i = i*16 + 8 - (i==0)*8
            end_i = start_i + 16 + (i==0)*8 + (i==30)*8
           
            start_j = j*16 + 8 - (j==0)*8
            end_j = start_j + 16 + (j==0)*8 + (j==30)*8

            take_si = 8 - (i==0)*8
            take_ei = take_si + 16 + (i==0)*8 + (i==30)*8 

            take_sj = 8 - (j==0)*8
            take_ej = take_sj + 16 + (j==0)*8 + (j==30)*8


            if wv < 3:
                rgb[start_i: end_i, start_j: end_j,wv] = ip_arr[take_si:take_ei,take_sj:take_ej,wv]

            truth[start_i: end_i, start_j: end_j,wv] = op_arr[take_si:take_ei,take_sj:take_ej, wv]

            u_pred[start_i: end_i, start_j: end_j,wv] = unet[index,take_si:take_ei,take_sj:take_ej, wv]
            r_pred[start_i: end_i, start_j: end_j,wv] = resnet[index,take_si:take_ei,take_sj:take_ej, wv]
        
        if wv < 3:
            rgb[:, :, wv] = rgb[:, :, wv]
        u_pred[:, :, wv] = u_pred[:, :, wv]
        u_mrae[:, :, wv] = u_mrae[:, :, wv]
        r_pred[:, :, wv] = r_pred[:, :, wv]
        r_mrae[:, :, wv] = r_mrae[:, :, wv]
        truth[:, :, wv] = truth[:, :, wv]

    truth[truth == 0] = 0.0001
    u_mrae = np.abs(u_pred - truth) / truth;
    r_mrae = np.abs(r_pred - truth) / truth;
    print("RESNET MRAE: ", np.mean(u_mrae))
    print("DEEPERESNET MRAE: ", np.mean(r_mrae))
    u_mrae = np.abs(u_pred - truth);
    r_mrae = np.abs(r_pred - truth);
    print(np.mean(u_mrae))
    print(np.mean(r_mrae))
    u_mrae[u_mrae > 2] = 2
    r_mrae[r_mrae > 2] = 2

    
    for wv in range(31):
        
        print(f"Wavelength: {wv}....")
        fig, axs = plt.subplots(2, 3)
        axs[0, 2].imshow(truth[:,:,wv], cmap = 'gist_gray')
        axs[0, 2].set_title('Truth Value')
        axs[0, 2].set_axis_off()
        axs[1, 2].imshow(rgb)
        axs[1, 2].set_axis_off()
        

        axs[0, 0].imshow(u_pred[:,:,wv], cmap = 'gist_gray')
        axs[0, 0].set_title('Deeper ResNet Pred')        
        axs[0, 0].set_axis_off()
        
        axs[0, 1].imshow(r_pred[:,:,wv], cmap = 'gist_gray')
        axs[0, 1].set_title('Resnet Pred')
        axs[0, 1].set_axis_off()
        
        ue = axs[1, 0].imshow(u_mrae[:,:,wv], cmap = 'jet')
        axs[1, 0].set_title('Deeper ResNet Error')
        axs[1, 0].set_axis_off()
        fig.colorbar(ue, ax = axs[1, 0])
        
        re = axs[1, 1].imshow(r_mrae[:,:,wv], cmap = 'jet')
        axs[1, 1].set_title('Resnet Error')
        axs[1, 1].set_axis_off()
        fig.colorbar(re, ax = axs[1, 1])
        
        
        fig.set_size_inches((11, 8.5), forward=False)
        fig.suptitle(f"Wavelength: {wv*10 + 400} nm, Image ID: 10")
        plt.savefig(f"/workspaces/ps_project/plots/test_{wv}.png", dpi=100)
        plt.close(fig)
    