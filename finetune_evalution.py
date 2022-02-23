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

#Dataset path
DATASET_PATH = '/home/ganesh/Documents/speech_recognition/data_sets/'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
num_labels = len(commands)
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

#spliting the data into train, test and validation in the ratios of 80:10:10
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

model = tf.keras.models.load_model('lstm.h5')
model.trainable = False
base_inputs = model.layers[0].input
base_outputs = model.layers[1].output
final_outputs = tf.keras.layers.Dense(14)(base_outputs)
model = tf.keras.Model(inputs = base_inputs, outputs = final_outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'],)
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),)
model.save('finetune_lstm.h5')

##model evaluation on test data
test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

#model accuracy calcualtion for the test data
model = tf.keras.models.load_model('finetune_lstm.h5')
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels
test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

#confusion matrix on the testdata
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

