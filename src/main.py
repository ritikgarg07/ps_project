import tensorflow as tf
import model
from datetime import datetime
import data
import time
import os
from benchmark import timeit
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


train_dataset = data.DataSet(batch_size=8, mode = 'train')
train_ds = train_dataset.load_data()

validation_dataset = data.DataSet(batch_size=8, mode = 'validation')
validation_ds = validation_dataset.load_data()

test_dataset = data.DataSet(batch_size = 8, mode = 'test')
test_ds = test_dataset.load_data()

unet = model.unet()
logdir = '/workspaces/ps_project/logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_dir = '/workspaces/ps_project/models/train/'
checkpoint_path = '/workspaces/ps_project/models/train/cp-{epoch:04d}.ckpt'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

# unet.load_weights(latest_checkpoint)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1,period=5)

history = unet.fit(train_ds, validation_data=validation_ds,batch_size = 8, epochs=20, callbacks=[tensorboard_callback, cp_callback])

prediction = unet.predict(test_ds)
data.convert_image(prediction)

performance = unet.evaluate(test_ds)
