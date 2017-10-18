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
import Constants as cc
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
dataset = pandas.read_csv(cc.iris_url, names = cc.names)

# Get some information of dataset
print("\nData Shape is: ", dataset.shape)

# Get the first 20 rows of data
print("\nLet's take a look at our data\n")
print(dataset.head(20))

# Describe the dataset
print("\nLet's describe this dataset\n")
print(dataset.describe())

# Get the class distribution of this dataset
print(dataset.groupby('class').size())
