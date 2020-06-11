import data
import model
import os
import tensorflow as tf
import tensorboard
import time
import numpy as np
from PIL import Image

from datetime import datetime
from benchmark import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train_dataset = data.DataSet(batch_size=8, mode = 'train')
train_ds = train_dataset.load_data()

validation_dataset = data.DataSet(batch_size=8, mode = 'validation')
validation_ds = validation_dataset.load_data()

test_dataset = data.DataSet(batch_size = 8, mode = 'test')
test_ds = test_dataset.load_data()

unet = model.unet()
logdir = '/workspaces/ps_project/logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# history = unet.fit(train_ds, validation_data=validation_ds,batch_size = 8, epochs=10, callbacks=[tensorboard_callback])

prediction = unet.predict(test_ds)
data.convert_image(prediction)



