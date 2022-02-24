# Speech-command-classfication
This repository presents LSTM basd model designed to identify keywords in short segments of audio. It has been tested using the [Google Speech Command Datasets](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). 
## Prerequsits:
We have usde python 3.8 with following packages 
```python
pip install numpy
pip install h5py
pip install seaborn
pip install tensorflow==2.7
pip install keras
pip install matpotlib

```
## Dataset for Task1: 
The training samples are taken from the [Google Speech Command Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). 10 classes are 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'. Two other classes named ‘silence’ and ‘unknown’ should also be predicted in case of a silent audio clip or an unknown keyword.
(A total of 12 classes including keywords).
### Dataset for the silent class: 
For silent class, we have taken speech samples for noise from the google speech command dataset. To make the balanced data set for the each class, we have taken number of silent class equal to average of remining class average number of class samples.

### Dataset for the unknown class:
For unknow class model is able to detect the word not in the list of classes, to make balanced class labels audio files randomly choosen from the other than the classes present in the speech command dataset. Once the data set is ready then the prepared the dataset for train, test and valdiation in the  ratios of the 80:10:10 from the all classes.

### Network learning process:  

<img src="https://user-images.githubusercontent.com/100190176/155363067-29d29821-f4ea-4815-a4c3-d9ee1da86103.png" width="500" height="300">

The network takes the speech spectrogram as input and predicts one of the class labels. The cross entropy loss function used to minimize the error between the predicted values and actual class labels. Speech signals of non stationary nature and each frame of the spectrogram highly correlated with neighbouring frames, the dependencies are well captured if use of recurrent neural network for the model prediction. For this we have used the LSTM with 250 nodes as first layers and followed by feed forward layers with relu activation function. The final layer should be able to give the class probabilities of all classes, we have used the softmax layer as the final output layer with the number of nodes equal to the number classes.

### Network architecture:
<img src="https://user-images.githubusercontent.com/100190176/155454713-180b355c-59f1-40e7-81e3-119d25ec4879.png" width="500" height="300">

### Training Vs Validation loss:


<img src="https://user-images.githubusercontent.com/100190176/155456384-157b2078-0f2d-4195-ae46-bcc13c3b38c7.png" width="500" height="300">




### Confusion Matrix for Test data:


<img src="https://user-images.githubusercontent.com/100190176/155456410-d0b1418c-7cda-4d09-ba3a-4fc09017e2cb.png" width="500" height="300">





### Network architecture for fine tuning:

<img src="https://user-images.githubusercontent.com/100190176/155456384-157b2078-0f2d-4195-ae46-bcc13c3b38c7.png" width="500" height="300">


### Training Vs Validation loss:

<img src="https://user-images.githubusercontent.com/100190176/155456410-d0b1418c-7cda-4d09-ba3a-4fc09017e2cb.png" width="500" height="300">



### Confusion Matrix for Test data:


<img src="https://user-images.githubusercontent.com/100190176/155454713-180b355c-59f1-40e7-81e3-119d25ec4879.png" width="500" height="300">
