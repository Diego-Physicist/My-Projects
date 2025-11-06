# This tutorial was extracted from:
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# Thanks for the help learning ML

# Step 1: First load the libraries

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Step 2: pandas is used to load the data

# Names of the atributes
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# We set the atributes names to the corresponding columns in iris.data
dataset = read_csv("iris.data", names=names)

# We could do this previous step using an URL:
#
#   url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
#   dataset = read_csv(url, names=names)

# Step 3: Now summarize the dataset in four different ways

# A) shape
print(dataset.shape) # (150, 5)

# B) head
print(dataset.head(20)) # 20 first rows from the iris.data

# C) descriptions
print(dataset.describe()) # Statistical properties of the iris.data

# D) class distribution
print(dataset.groupby('class').size()) # It displays the clases and the amount of flowers of each class

# Step 4: Data Visualization with two types of plots

# A) Univariate Plots --> better understanding of each atribute

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()  # The sepal length and width approach a gaussian distribution

# B) Multivariate plots --> better understanding of the relationships between atributes

# scatter plot matrix
scatter_matrix(dataset) # Note the diagonal grouping of some pairs of attributes. This could be used to make predictions
plt.show()

# Step 5: Now we evaluate some algorithms by creating models to fit the data and estimate their accuracy on unseen data

# A) We will split the loaded dataset into two, 80% of which we will use to train, evaluate and select among our models, and 20% that we will hold back as a validation dataset.
# Split-out validation dataset
array = dataset.values
X = array[:,0:4] # First 4 attributes
y = array[:,4] # Class attribute
# train_test_split separates the date randomly (in this case the random_state=1 means that it's the same seed for every event) in training part and test part
# test_size can be a percentage or an int meaning the number of data samples
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# B) Test Harness: We will use stratified 10-fold cross validation to estimate model accuracy.
#                  This will split our dataset into 10 parts, train on 9 and test on 1 and repeat
#                  for all combinations of train-test splits(k-Fod cross-validation)

# More info: https://machinelearningmastery.com/k-fold-cross-validation/

# The accuracy metric is the ratio between the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate).

# C) Build models: let's try several algorithms to see which one is the best

# Logistic Regression (LR) --> LINEAR
# Linear Discriminant Analysis (LDA) --> LINEAR
# K-Nearest Neighbors (KNN) --> NONLINEAR
# Classification and Regression Trees (CART) --> NONLINEAR
# Gaussian Naive Bayes (NB) --> NONLINEAR
# Support Vector Machines (SVM) --> NONLINEAR

# Spot Check Algorithms ( I'm not going to explain each model here )
models = []
# Use modern solver/settings for multiclass (Iris has 3 classes); in the example is different
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200, random_state=1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=1)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# The last prints show that  it looks like Support Vector Machines (SVM) has the largest
# estimated accuracy score at about 0.98 or 98%. But we can create a plot of the model evaluation
# results and compare the spread and mean accuracy of each model.

# Compare Algorithms. The results array contains an array of scores of the estimator for each run of the cross validation and for each model in this case.
plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.show()

# We can see that the box and whisker plots are squashed at the top of the range, with many
# evaluations achieving 100% accuracy, and some pushing down into the high 80% accuracies.

# Step 6: We need to choose an algorithm to make predictions, so we are choosing SVM

# Make predictions on validation dataset
model = SVC(gamma='auto') # We choose SVM model
model.fit(X_train, Y_train) # Then we fit the model with the trainning arrays
predictions = model.predict(X_validation) # Finally we make predictions with the testing array X

# In the end, we evaluate the predictions comparing those to the Y_validation array
print("\n" + "="*50)
print("🔹 MODEL EVALUATION RESULTS 🔹")
print("="*50)

print(f"✅ Accuracy Score: {accuracy_score(Y_validation, predictions):.4f}\n")

print("📊 Confusion Matrix:")
print(confusion_matrix(Y_validation, predictions))
print("\n🧾 Classification Report:")
print(classification_report(Y_validation, predictions))
print("="*50)

# We can see that the accuracy is 0.9667 or about 96% on the hold out dataset. The confusion
# matrix provides an indication of the errors made. Finally, the classification report
# provides a breakdown of each class by precision, recall, f1-score and support showing
# excellent results (granted the validation dataset was small).