import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix

DIR = r'C:\Users\A747043\Desktop\My documents\Python\Python36\PyCharm Projects\Deep Learning\ANN'

# Import dataset
dataset = pd.read_csv(DIR + '\Churn_Modelling.csv')

### Dataset features
# First 3 are irrelevant 'RowNumber', 'CustomerId', 'Surname'
# Last is the classification variable Y
print(dataset.keys())

# 'Geography' & 'Gender' are categorical variables
print(dataset.head())

# Allocate dataset to X and y variables
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

### Categorical variables
# LabelEncoder transforms categorical values to numerical values
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

# OneHotEncoder used to prevent misleading classification by the algorithm, such as 0 < 1 < 2
# Specify which column to OneHotEncode: categorical_features=[1]
onehotencoder = OneHotEncoder(categorical_features=[1])

# OneHotEncoder returns a sparse matrix. Using .toarray() allows to work with easier format.
# Alternatively, canse use sparse=False as argument in onehotencoder above
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Split dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

### Feature scaling: set all on same scale prevents model giving more importance to some features: # e.g. age vs  salary (hundred vs thousands)
# fit : gets mean and variance | transform: transform all values, substract mean, divide by variance

# Use fit from train, to make sure we scale them according to same mean and variances. Allows to standardize training
# data to any test set. Test set could have different distribution than train, and could bias the model because of
# test data.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Building the ANN
# Initialize ANN
classifier = Sequential()

# First input layer and hidden layer
# 11 input nodes, 6 output nodes (here, 'hidden nodes')
# Kernel_initializer : Initializations define the way to set the initial random weights of Keras layers.
# Activation function transforms the summed weighted input from the node into the activation of the node or output for that input
classifier.add(Dense(input_dim=11, kernel_initializer='uniform', activation= 'relu', units=6))

# Second layer
classifier.add(Dense(kernel_initializer='uniform', activation='relu', units=6))

# Third layer
classifier.add(Dense(kernel_initializer='uniform', activation='relu', units=6))

# Output layer
classifier.add(Dense(kernel_initializer='uniform', activation='sigmoid', units=1))

# Compile ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit ANN to training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

### Predict a single individual
"""Will this customer leave the bank?
Geography: Spain
Credit Score: 650
Gender: Male
Age: 55
Tenure: 2
Balance: 65000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 60000"""

# Use double [[ ]] because we want an horizontal array
# Geography: 3 countries possible. 0.0, 0.0 is France, 0.0, 1.0 is Spain, 1.0, 0.0 Germany. Other variables are clear.
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 1.0, 650, 1, 55, 2, 650000, 2, 1, 1, 60000]])))
new_prediction = (new_prediction > 0.5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluating the ANN with cross_validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# n_jobs = -1 means all CPUs will be used
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# Given the possible variance between results we take the average of the accuracies
mean = accuracies.mean()
variance = accuracies.std()

## Results:
# Mean of accuracies 0.8394999952986837
# Variance of accuracies 0.018991774234018002

### Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Best parameters turn out to be: batch_size 25, epochs 500, optimizer rmsprop
