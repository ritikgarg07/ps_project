import os
import tensorflow as tf
import yaml
from datetime import datetime

import data
import model
from benchmark import timeit
import sample_prepare
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading config gile
with open('/workspaces/ps_project/config.yaml') as file:
    config = yaml.safe_load(file)

# Create dataset from images in sample/
prepare = sample_prepare.SamplePrepare(dest_path = config["base_dir"] + 'sample/', src_path = config["base_dir"] + 'sample/', size = config["patch_size"])
prepare.prepare_sample()

# Create tf object for dataset
sample_dataset = data.DataSet(batch_size = config["batch_size"], mode = 'sample', path = config["base_dir"] + 'sample/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
sample_ds = sample_dataset.load_data()
print("\nDataset created and loaded")

# Model setup
checkpoint_dir_u = config["base_dir"] + 'models/unet/train/'
checkpoint_path_u = config["base_dir"] + 'models/unet/train/cp-{epoch:04d}.ckpt'
latest_checkpoint_u = tf.train.latest_checkpoint(checkpoint_dir_u)

checkpoint_dir_r = config["base_dir"] + 'models/resnet/train/'
checkpoint_path_r = config["base_dir"] + 'models/resnet/train/cp-{epoch:04d}.ckpt'
latest_checkpoint_r = tf.train.latest_checkpoint(checkpoint_dir_r)

checkpoint_dir_r2 = config["base_dir"] + 'models/resnet2/train/'
checkpoint_path_r2 = config["base_dir"] + 'models/resnet2/train/cp-{epoch:04d}.ckpt'
latest_checkpoint_r2 = tf.train.latest_checkpoint(checkpoint_dir_r2)

unet = model.unet(input_size=(config["patch_size"], config["patch_size"], 3), wavelengths= config["wavelengths"])
resnet = model.resnet(input_dim=config["patch_size"], wavelengths = config["wavelengths"])
resnet2 = model.resnet2(input_dim=config["patch_size"], wavelengths = config["wavelengths"])


# unet.load_weights(latest_checkpoint_u)
# print("\nLoaded pre-trained model")
# prediction_u = unet.predict(test_ds)
# print("Prediction completed (using unet model)")

resnet.load_weights(latest_checkpoint_r)
print("\nLoaded pre-trained model")
prediction_r = resnet.predict(sample_ds)
print("\nPrediction completed (using resnet model)")

# resnet2.load_weights(latest_checkpoint_r2)
# print("\nLoaded pre-trained model")
# prediction_r2 = resnet2.predict(sample_ds)
# print("Prediction completed (using deeper resnet model)")

print("\nStoring result in ps_project/results/")
data.convert_image(prediction_r, config["wavelengths"], config["patch_size"], config["image_size"])
