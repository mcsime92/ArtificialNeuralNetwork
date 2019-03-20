import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential

# Import dataset
dataset = pd.read_csv(r"C:\Users\A747043\Desktop\My documents\Python\PyCharm Projects\Deep Learning\own_models\ANN\Churn_Modelling.csv")

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
