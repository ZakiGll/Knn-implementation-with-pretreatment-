import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from knn import knn

adult = pd.read_csv('horse_colic.csv', na_values=['?'])
df = pd.DataFrame(adult)
X = df.iloc[:, :-1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
y = df.iloc[:, -1]
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = knn(5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print("Le resultats avant le prétraitement: ")
acc = np.sum(prediction == y_test) / len(y_test)
print(acc)

df = df.fillna(df.mean(numeric_only=True))

X = df.drop(columns=['class'])
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

selector = SelectKBest(mutual_info_classif, k=5)
selector.fit(X, y)

relevant_features_mask = selector.get_support()
relevant_features = X.columns[relevant_features_mask]
X = X[relevant_features]

n_bins = 4
for col in X.columns:
    X[col] = pd.cut(X[col], n_bins, labels=False)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf = knn(5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print("Le resultats apres le prétraitement: ")
acc = np.sum(prediction == y_test) / len(y_test)
print(acc)
