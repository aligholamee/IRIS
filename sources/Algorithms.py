# IRIS Classification 
# ========================================
# [] File Name : Algorithms.py
#
# [] Creation Date : October 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

model = []
model.append(("LR", LogisticRegression()))
model.append(("LDA", LinearDiscriminantAnalysis()))
model.append(("KNN", KNeighborsClassifier()))
model.append(("CART", DecisionTreeClassifier()))
model.append(("NB", GaussianNB()))
model.append(("SVM", SVC()))
