import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
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

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
predicted = knn.predict(x_val)

param_range = np.arange(1, 10, 1)

train_score, test_score = validation_curve(knn,
                                           X=x_train, y=y_train, cv=4, n_jobs=10, param_name="n_neighbors",
                                           param_range=param_range)

mean_train_score = np.mean(train_score, axis=1)
std_train_score = np.std(train_score, axis=1)

# Calculating mean and standard deviation of testing score
mean_test_score = np.mean(test_score, axis=1)
std_test_score = np.std(test_score, axis=1)

# Plot mean accuracy scores for training and testing scores
plt.plot(param_range, mean_train_score,
         label="Training Score", color='b')
plt.plot(param_range, mean_test_score,
         label="Cross Validation Score", color='g')

# Creating the plot
plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

scores = cross_val_score(knn, x_val, y_val, cv=5)
print('scores per fold ', scores)
print('mean score    ', np.mean(scores))
print('standard dev. ', np.std(scores))

print(classification_report(y_val, predicted))

confusion_matrix = plot_confusion_matrix(knn, x_val, y_val, normalize='true', values_format='.2f')
confusion_matrix.figure_.suptitle("K-nearest Neighbours Confusion Matrix")
plt.show()
