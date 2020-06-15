import os
import tensorflow as tf
import yaml
from datetime import datetime

import data
import model
from benchmark import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ! Specifiy absolute path here
with open('/workspaces/ps_project/config.yaml') as file:
    config = yaml.safe_load(file)


train_dataset = data.DataSet(batch_size = config['batch_size'], mode = 'train', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
train_ds = train_dataset.load_data()

validation_dataset = data.DataSet(batch_size = config["batch_size"], mode = 'validation', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
validation_ds = validation_dataset.load_data()

test_dataset = data.DataSet(batch_size = config["batch_size"], mode = 'test', path = config["base_dir"] + 'data/', patch_size=config["patch_size"], wavelengths=config["wavelengths"])
test_ds = test_dataset.load_data()


unet = model.unet(input_size=(config["patch_size"], config["patch_size"], 3), wavelengths= config["wavelengths"])


logdir = config["base_dir"] + 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = config["base_dir"] + 'models/train/'
checkpoint_path = config["base_dir"] + 'models/train/cp-{epoch:04d}.ckpt'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,save_best_only=5)


unet.load_weights(latest_checkpoint)
# history = unet.fit(train_ds,validation_data= validation_ds, batch_size = config["batch_size"], epochs = config["epochs"], callbacks=[tensorboard_callback, cp_callback])

prediction = unet.predict(test_ds)
data.plot_spectrum_by_wv(prediction)
data.plot_spectrum_by_pixel(prediction)
# data.convert_image(prediction)
# performance = unet.evaluate(test_ds)
