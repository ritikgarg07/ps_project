import tensorflow as tf
import model
from datetime import datetime
import DataSet
import time
import tensorboard
import os
from benchmarks import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


dataset = DataSet.DataSet(batch_size=8)
train_ds = dataset.load_process(train=True)
# train_ds = dataset.load_process(train=True)
# validation_ds = dataset.load_process(validation=True)

unet = model.unet()
logdir = '/workspaces/ps_project/logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# for f in test_ds.take(5):
#     print(f[1])
#     # pass

timeit(dataset)



# history = unet.fit(train_ds, validation_data = validation_ds, batch_size = 8, epochs=2, callbacks=[tensorboard_callback])
