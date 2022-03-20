import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from pca import digits_train_pca, digits_test_pca, x_train, y_train, x_test
from file_output import output_file

# digits_test = pd.read_csv('data/test.csv')

# x_train = digits_train_pca.iloc[:, 1:]
# y_train = digits_train_pca.loc[:, 'label']
#
# x_test = digits_test_pca

# print(len(x_train))
# print(len(y_train))
#
# print(len(x_val))
# print(len(y_val))

# param_grid = {
#     'C': [0.1, 1, 10, 100, 1000],
#     'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
#     'kernel': ['rbf', 'sigmoid']
# }

svm = SVC(C=10)
# grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=10, pre_dispatch=10)
# grid_search = grid.fit(x_train, y_train)
#
# print(grid.best_estimator_)
# print(grid.best_params_)
# accuracy = grid_search.best_score_ * 100
# print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))

svm.fit(x_train, y_train)
predicted_y_test = svm.predict(x_test)

output_file("results/svm.csv", predicted_y_test)

# scores = cross_val_score(svm, x_val, y_val, cv=5)
# print('scores per fold ', scores)
# print('mean score    ', np.mean(scores))
# print('standard dev. ', np.std(scores))
#
# print(classification_report(y_val, predicted))
#
# confusion_matrix = plot_confusion_matrix(svm, x_val, y_val, normalize='true', values_format='.2f')
# confusion_matrix.figure_.suptitle("Support Vector Machine Confusion Matrix")
# plt.show()
