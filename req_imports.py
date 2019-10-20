

############################################## General Imports ###############################################

import os
os.environ['PYTHONHASHSEED']=str(100)
import time
import datetime
import pandas as pd
import os.path
import pickle
import numpy as np 
np.random.seed(100)

import math as m
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from pylab import plt
plt.style.use('seaborn')
get_ipython().magic(u'matplotlib inline')
from pandas_datareader import data as web





#
#os.environ["PATH"] += os.pathsep + 'C:/Program Files/Anaconda3/Library/bin/graphviz/'

############################################## ML Imports ###############################################

from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import tensorflow as tf
tf.random.set_random_seed(100)
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.feature_selection import SelectKBest, f_classif
from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import tree
#import graphviz

from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from keras.utils import to_categorical