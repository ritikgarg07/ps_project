import tensorflow as tf
import data  
import model

train_path = '/workspaces/ps_project/data/train/'

# List of input images in given folder defined by path variable
list_ds_train = tf.data.Dataset.list_files(str(train_path + '*.bmp'))

# List of tuples (input, output) in given folder defined by path variable
labelled_ds_train = list_ds_train.map(data.process_ds, num_parallel_calls=data.AUTOTUNE)
labelled_ds_train = data.load_process(labelled_ds_train)

validation_path = '/workspaces/ps_project/data/validation/'
# List of input images in given folder defined by path variable
list_ds_validation = tf.data.Dataset.list_files(str(validation_path + '*.bmp'))

# List of tuples (input, output) in given folder defined by path variable
labelled_ds_validation = list_ds_validation.map(data.process_ds, num_parallel_calls=data.AUTOTUNE)
labelled_ds_validation = data.load_process(labelled_ds_validation)


unet = model.unet()
history = unet.fit(labelled_ds_train, epochs=2)