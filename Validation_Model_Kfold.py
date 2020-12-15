# k-Fold Cross Validation
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from numpy import mean
from numpy import std

# Importing the dataset
dataset = pd.read_csv('iris.csv')
X = dataset.iloc[:, [0, 1, 2, 3]].values
y = dataset.iloc[:, 4].values
print('Data X')
print(X)
print('Data Y')
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# prepare the cross-validation procedure (K=10)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
# create model
model = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X = X_train, y = y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy K-Fold Model: %.3f (%.3f)' % (mean(scores), std(scores)))

#Klasifikasi KNN (k=5)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

#Hasil prediksi data testing menggunakan KNN
hasil_prediksi_out = classifier.predict(X_test)
print('Data Tes')
print(y_test)
print('Data Prediksi')
print(hasil_prediksi_out)
print(classification_report(y_test, hasil_prediksi_out))

