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
pip install mathpotlib

```
## Dataset: 
The training samples are taken from the [Google Speech Command Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html). 10 classes are 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'. Two other classes named ‘silence’ and ‘unknown’ should also be predicted in case of a silent audio clip or an unknown keyword.
(A total of 12 classes including keywords).
### Dataset for the silent class: 
For silent class, we have taken speech samples for noise from the google speech command dataset. To make the balanced data set for the each class, we have taken number of silent class equal to average of remining class average number of class samples.

### Dataset for the unknown class:
