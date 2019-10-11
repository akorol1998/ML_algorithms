

import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("actual_data.txt")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True) # Here we are dropping the column with name "id"

# We can also drop rows by index df.drop([0, 1]) - will drop rows with index 1 and 2



X = np.array(df.drop(['class'], 1)) # Droping class column, and leaving everything else
y = np.array(df['class'])

print(df)
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)



# Actually have to search for some data manipulation tutorials to become efficient at using reshape() or shape()

examle_measure = np.array([4,8,9,5,6,2,9,9,1]).reshape(1,-1)
prediction = clf.predict(examle_measure)

print(prediction)
