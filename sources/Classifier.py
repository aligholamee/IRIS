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

# Load algorithms
import Algorithms as algo 

# Load libraries | Tools
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

# Display the box and whiskers plots
# dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show()

# Display the histogram diagram
# dataset.hist()
# plt.show()

# scatter_matrix(dataset)
# plt.show()

splitter_array = dataset.values
print("\n\n\n\n")
X = splitter_array[:, 0:4]
Y = splitter_array[:, 4]

ValidationSize = 0.2
seed = 7
X_Train, X_Validation, Y_Train, Y_Validation = model_selection.train_test_split(X, Y, test_size=ValidationSize, random_state=seed)

# Evaluate the models(algorithms) 
results = []
names = []

for name, model in algo.models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_Train, Y_Train, cv=kfold, scoring=cc.scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Display the comparison plots
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


