import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import glob
import cv2
import sklearn
import skimage
import keras

from sklearn import metrics, svm, multiclass
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import roc_curve, auc

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.models import load_model
from keras.models import clone_model
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.optimizers import SGD

from skimage import data, img_as_float
from skimage import exposure
from skimage import data, img_as_float
from skimage import exposure

import random
import time

from copy import deepcopy
from collections import Counter

#Disable annoying TensorFlow warnings
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)