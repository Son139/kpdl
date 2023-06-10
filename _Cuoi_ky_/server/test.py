import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('purchase_history.csv')

print(df.head())
print(df.count())

gender_encoded = pd.get_dummies(df['Gender'], drop_first=True, dtype=int)
print(gender_encoded)

df = pd.concat([df,gender_encoded],axis=1)
print(df)

x = df[['Male','Age','Salary','Price']].to_numpy()
print(x)

y = df['Purchased'].to_numpy()
print(y)

print("----------------------------------------------------")

x_train , x_test , y_train , y_test = train_test_split(x,y, test_size=0.2, random_state=20)
# print("------x_train: " , x_train)
# print("------x_test: " ,x_test)

print(len(x_train), len(x_test), len(y_train), len(y_test))

print("----------------------------------------------------")

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print("----------------------------------------------------KNN")

k= 5
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(x_train,y_train)
KNeighborsClassifier()

y_pred = knn.predict(x_test)
print(y_pred)
print(y_test)

print("----------------------------------------------------accurary")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("----------------------------------------------------save model")

# with open('./models/knn_model.pickle','wb') as f:
#   pickle.dump(knn,f)
  
# with open('./models/scaler.pickle','wb') as f:
#   pickle.dump(scaler,f)

