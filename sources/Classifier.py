# IRIS Classification 
# ========================================
# [] File Name : Classifier.py
#
# [] Creation Date : October 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#

# Load Constants
from Constants import *
# Load libraries | Tools
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Initialise the dataset
dataset = pandas.read_csv(Constants.url, names=Constants.names)
