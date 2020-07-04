import csv
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

train_df = pd.read_csv('mnist_train.csv')
test_df = pd.read_csv('mnist_test.csv')

y_train = train_df[['label']]
train_df.drop(['label'], axis = 1, inplace = True)
X_train = train_df[:]


y_test = test_df[['label']]
test_df.drop(['label'], axis = 1, inplace = True)
X_test = test_df[:]


#train_df.info()

#test_df.info()

#print(y_train[0:10])

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)

clf = MLPClassifier(solver='adam', hidden_layer_sizes = (100,), activation='tanh', alpha = 0.0001)

clf.fit(X_train, y_train)
neural_output = clf.predict(X_test)
print("sgd")
print(accuracy_score(y_test,neural_output))

#clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), random_state=1)
#clf.fit(X_train, y_train)   
#neural_output = clf.predict(X_test)
#print("sgd")
#print(accuracy_score(y_test, neural_output))





