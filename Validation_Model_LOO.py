import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_excel('G:/A S2/SEM 1/AI_Pak Ali Ridho/M7/iris.xlsx')

x = data.drop(columns=['species'])
y = data['species']

loo = LeaveOneOut()
#loo.get_n_splits(x)

true_y, pred_y = list(), list()
for train_index, test_index in loo.split(x):
    train_x, test_x = x.iloc[train_index, :], x.iloc[test_index, :]
    train_y, test_y = y.iloc[train_index], y.iloc[test_index]

#KNN Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)
distances, indexs=classifier.kneighbors(test_x,return_distance=True)

# evaluate model
yhat = classifier.predict(test_x)
yhatt=yhat.tolist()
test_yy=test_y.tolist()
    
# store
true_y.append(test_yy[0])
pred_y.append(yhatt[0])

print(distances)
print(indexs)
print(true_y)
print(pred_y)
print(classification_report(true_y, pred_y))

# calculate accuracy
acc = accuracy_score(true_y, pred_y)
print('Accuracy: %.3f' % acc)
