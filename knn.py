import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from pca import digits_train_pca, digits_test_pca
from file_output import output_file

# digits_test = pd.read_csv('data/test.csv')

x_train = digits_train_pca.iloc[:, 1:]
y_train = digits_train_pca.loc[:, 'label']

x_test = digits_test_pca.iloc[:, 1:]

# print(x_train)
# print(x_train.shape)
#
# print(x_test)
# print(x_test.shape)

knn = KNeighborsClassifier(7)
knn.fit(x_train, y_train)
predicted_y_test = knn.predict(x_test)

print(predicted_y_test)
print(predicted_y_test.shape)

output_file("results/knn.csv", predicted_y_test)

# scores = cross_val_score(knn, x_val, y_val, cv=5)
# print('scores per fold ', scores)
# print('mean score    ', np.mean(scores))
# print('standard dev. ', np.std(scores))
#
# print(classification_report(y_val, predicted))
#
# confusion_matrix = plot_confusion_matrix(knn, x_val, y_val, normalize='true', values_format='.2f')
# confusion_matrix.figure_.suptitle("K-nearest Neighbours Confusion Matrix")
# plt.show()
