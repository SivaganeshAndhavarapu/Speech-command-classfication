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


sample_file ='/home/ganesh/Documents/speech_recognition/data_sets/silent/noise_2100.wav'
model = tf.keras.models.load_model('model_lstm.h5')

DATASET_PATH = '/home/ganesh/Documents/speech_recognition/data_sets/'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))

AUTOTUNE = tf.data.experimental.AUTOTUNE
def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map( map_func=get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map( map_func=get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
  return output_ds

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

sample_ds = preprocess_dataset([str(sample_file)])
for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  print('----------------------------------------')
  print("The predicted label is :",commands[label[0]])
  print('----------------------------------------')
