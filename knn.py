import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from pca import digits_pca

# digits = pd.read_csv('data/train.csv')

x = digits_pca.iloc[:, 1:]
y = digits_pca.loc[:, 'label']


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

print(len(x_train))
print(len(y_train))

print(len(x_val))
print(len(y_val))

knn = KNeighborsClassifier(7)
knn.fit(x_train, y_train)
predicted = knn.predict(x_val)

scores = cross_val_score(knn, x_val, y_val, cv=5)
print('scores per fold ', scores)
print('mean score    ', np.mean(scores))
print('standard dev. ', np.std(scores))

print(classification_report(y_val, predicted))

confusion_matrix = plot_confusion_matrix(knn, x_val, y_val, normalize='true', values_format='.2f')
confusion_matrix.figure_.suptitle("K-nearest Neighbours Confusion Matrix")
plt.show()
