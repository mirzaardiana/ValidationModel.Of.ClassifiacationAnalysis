import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

#read file
data = pd.read_csv('G:/A S2/SEM 1/AI_Pak Ali Ridho/M7/iris.csv')

print(data.head)
#splitting 80% and 20% of first data from each label for train and test
train1=data[0:40]
test1=data[40:50]
train2=data[50:90]
test2=data[90:100]
train3=data[100:140]
test3=data[140:150]

#Concatenate train and test data
train=train1.append([train2,train3])
test=test1.append([test2,test3])

#optional : shuffle the data to erase order dependent anomaly in data
train=shuffle(train)
test=shuffle(test)

#removing label from train and test input
train_x = train.drop(columns=['species'])
test_x  = test.drop(columns=['species'])

#making class label as output
train_y = train['species']
test_y =  test['species']

# Evaluate using a train and a test set
model = LogisticRegression()
model.fit(train_x,train_y)
result = model.score(test_x,test_y)
print('Accuracy Hold-Out Cross Validation :')
print("Accuracy: %.2f%%" % (result*100.0))

#KNN Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_x, train_y)

#predict test data using KNN
pred_y = classifier.predict(test_x)
print(classification_report(test_y, pred_y))
print(pred_y)
print(test_y)
