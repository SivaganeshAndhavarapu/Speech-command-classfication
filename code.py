import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import h5py
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.models import Model
from feat_utils import decode_audio, get_label, get_waveform_and_label, get_spectrogram

DATASET_PATH = '/home/ganesh/Documents/speech_recognition/data_sets/'
data_dir = pathlib.Path(DATASET_PATH)

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
num_labels = len(commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
no_of_train_files=round(0.8*num_samples)
no_of_test_files=round(0.8*num_samples)
train_files = filenames[:no_of_train_files]
val_files = filenames[no_of_train_files: no_of_train_files + no_of_test_files]
test_files = filenames[-no_of_test_files:]

AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)

for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(map_func=get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size =256 
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape

norm_layer = tf.keras.layers.experimental.preprocessing.Normalization()
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

"""
model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model = models.Sequential([
    layers.Input(shape=(124,129)),
    layers.LSTM(129),
    layers.Dense(num_labels),])
model.summary()
"""


inputs=layers.Input(shape=(124,129))
x1=layers.LSTM(250)(inputs)
x2=layers.Dense(100,activation='relu')(x1)
x3=layers.Dense(50,activation='relu')(x2)
x4=layers.Dense(10,activation='relu')(x3)
outputs=layers.Dense(num_labels, activation='softmax')(x4)

model=Model(inputs=inputs,outputs=outputs)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'],)
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),)
#model.save('task1.h5')
print('all ok')
