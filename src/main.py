import os
import tensorflow as tf
import yaml
from datetime import datetime

import data
import model
from benchmark import timeit
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load config file
with open('/workspaces/ps_project/config.yaml') as file:
    config = yaml.safe_load(file)

# Load train, test and validation dataset
train_dataset = data.DataSet(batch_size = config['batch_size'], mode = 'train', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
train_ds = train_dataset.load_data()
print("Loading test dataset...")

validation_dataset = data.DataSet(batch_size = config["batch_size"], mode = 'validation', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
validation_ds = validation_dataset.load_data()
print("Loading validation dataset...")

test_dataset = data.DataSet(batch_size = config["batch_size"], mode = 'test', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
test_ds = test_dataset.load_data()
print("Loading test dataset...\n")

# Create log and checkpoint directories
logdir = config["base_dir"] + 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir_u = config["base_dir"] + 'models/unet/train/'
checkpoint_path_u = config["base_dir"] + 'models/unet/train/cp-{epoch:04d}.ckpt'
latest_checkpoint_u = tf.train.latest_checkpoint(checkpoint_dir_u)

checkpoint_dir_r = config["base_dir"] + 'models/resnet/train/'
checkpoint_path_r = config["base_dir"] + 'models/resnet/train/cp-{epoch:04d}.ckpt'
latest_checkpoint_r = tf.train.latest_checkpoint(checkpoint_dir_r)

checkpoint_dir_r2 = config["base_dir"] + 'models/resnet2/train/'
checkpoint_path_r2 = config["base_dir"] + 'models/resnet2/train/cp-{epoch:04d}.ckpt'
latest_checkpoint_r2 = tf.train.latest_checkpoint(checkpoint_dir_r2)

# Generating callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
cp_callback_u = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_u,save_weights_only=True,verbose=1,save_best_only=5)
cp_callback_r = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_r,save_weights_only=True,verbose=1,save_best_only=5)
cp_callback_r2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_r2,save_weights_only=True,verbose=1,save_best_only=5)

unet = model.unet(input_size=(config["patch_size"], config["patch_size"], 3), wavelengths= config["wavelengths"])
print("Created unet model")

resnet = model.resnet(input_dim=config["patch_size"], wavelengths = config["wavelengths"])
print("Created resnet model")

resnet2 = model.resnet2(input_dim=config["patch_size"], wavelengths = config["wavelengths"])
print("Created deeper resnet model\n")


# print("Using unet model")
# unet.load_weights(latest_checkpoint_u)
# print("Loaded pre-trained unet model")
# history = unet.fit(train_ds,validation_data= validation_ds, batch_size = config["batch_size"], epochs = config["epochs"])
# prediction_u = unet.predict(test_ds)
# performance = unet.evaluate(test_ds)


print("Using resnet model")
# resnet.load_weights(latest_checkpoint_r)
# print("Loaded pre-trained model")
history = resnet.fit(train_ds, validation_data = validation_ds, batch_size = config["batch_size"], epochs = config["epochs"], callbacks = [cp_callback_r])
prediction_r = resnet.predict(test_ds)
# performance = resnet.evaluate(test_ds)

# print("Using deeper resnet model")
# resnet2.load_weights(latest_checkpoint_r2)
# print("Loaded pre-trained deeper resnet model")
# history = resnet2.fit(train_ds, validation_data = validation_ds, batch_size = config["batch_size"], epochs = config["epochs"], callbacks = [cp_callback_r2])
# prediction_r2 = resnet2.predict(test_ds)
# performance = resnet2.evaluate(test_ds)

print("\nPrediction complete, storing result images in ps_project/results/")
data.convert_image(prediction_r, config["wavelengths"], config["patch_size"], config["image_size"])

print("\nStored output images, preparing plots in ps_project/plots/")
# data.save_results(prediction_r, prediction_r2, mrae = False)
data.save_results(prediction_r, mrae = False)


# ! Set mrae = True for plotting mrae, and set it to false for plotting absolute error