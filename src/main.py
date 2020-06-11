import tensorflow as tf
import model
from datetime import datetime
import data
import time
import tensorboard
import os
from benchmark import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train_dataset = data.DataSet(batch_size=8)
train_ds = train_dataset.load_data()

validation_dataset = data.DataSet(batch_size=8)
validation_ds = validation_dataset.load_data()

# train_ds = dataset.load_process(train=True)
# validation_ds = dataset.load_process(validation=True)

unet = model.unet()
logdir = '/workspaces/ps_project/logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = unet.fit(train_ds, validation_data=validation_ds, batch_size = 8, epochs=20, callbacks=[tensorboard_callback])


