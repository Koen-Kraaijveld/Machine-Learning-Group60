import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, roc_curve, auc

digits = pd.read_csv('data/train.csv')

x_train = digits.iloc[0:30000, 1:]
x_val = digits.iloc[30000:, 1:]

y_train = digits.loc[0:29999, 'label']
y_val = digits.loc[30000:, 'label']

print(len(x_train))
print(len(y_train))

print(len(x_val))
print(len(y_val))

knn = KNeighborsClassifier(50)
knn.fit(x_train, y_train)
predicted = knn.predict(x_val)

# # plt.scatter(x, y, alpha=0.1)
# plt.plot(x[3:10], knn.predict(x[3:10]), label='knn 10')
# plt.show()
# print('MSE knn', mean_squared_error(y, knn.predict(x)))

print(sklearn.metrics.classification_report(y_val, predicted))

confusion_matrix = sklearn.metrics.plot_confusion_matrix(knn, x_val, y_val)
plt.show()
