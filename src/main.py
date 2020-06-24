import os
import tensorflow as tf
import yaml
from datetime import datetime

import data
import model
from benchmark import timeit
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ! Specifiy absolute path here
with open('/workspaces/ps_project/config.yaml') as file:
    config = yaml.safe_load(file)

train_dataset = data.DataSet(batch_size = config['batch_size'], mode = 'train', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
train_ds = train_dataset.load_data()

validation_dataset = data.DataSet(batch_size = config["batch_size"], mode = 'validation', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
validation_ds = validation_dataset.load_data()

test_dataset = data.DataSet(batch_size = config["batch_size"], mode = 'test', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
test_ds = test_dataset.load_data()

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

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
cp_callback_u = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_u,save_weights_only=True,verbose=1,save_best_only=5)
cp_callback_r = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_r,save_weights_only=True,verbose=1,save_best_only=5)
cp_callback_r2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_r2,save_weights_only=True,verbose=1,save_best_only=5)


# unet = model.unet(input_size=(config["patch_size"], config["patch_size"], 3), wavelengths= config["wavelengths"])
resnet = model.resnet(input_dim=config["patch_size"], wavelengths = config["wavelengths"])
resnet2 = model.resnet2(input_dim=config["patch_size"], wavelengths = config["wavelengths"])


# unet.load_weights(latest_checkpoint_u)
# history = unet.fit(train_ds,validation_data= validation_ds, batch_size = config["batch_size"], epochs = config["epochs"])
# prediction_u = unet.predict(test_ds)
# performance = unet.evaluate(test_ds)


resnet.load_weights(latest_checkpoint_r)
# history = resnet.fit(train_ds, validation_data = validation_ds, batch_size = config["batch_size"], epochs = config["epochs"], callbacks = [cp_callback_r])
prediction_r = resnet.predict(test_ds)
performance = resnet.evaluate(test_ds)


resnet2.load_weights(latest_checkpoint_r2)
# history = resnet2.fit(train_ds, validation_data = validation_ds, batch_size = config["batch_size"], epochs = config["epochs"], callbacks = [cp_callback_r2])
prediction_r2 = resnet2.predict(test_ds)
performance = resnet2.evaluate(test_ds)


data.convert_image(prediction_r2, config["wavelengths"], config["patch_size"], config["image_size"])
# data.plot_spectrum_by_wv(prediction_r, config["wavelengths"], config["patch_size"], config["image_size"], patch = 128)
# data.plot_spectrum_by_pixel(prediction_r, config["patch_size"], config["image_size"], patch = 128, x_s = 0, y_s = 0)
# data.save_results(prediction_r2, prediction_r)
