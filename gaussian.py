import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from pca import digits_train_pca, digits_test_pca
from file_output import output_file

# digits_test = pd.read_csv('data/test.csv')

x_train = digits_train_pca.iloc[:, 1:]
y_train = digits_train_pca.loc[:, 'label']

x_test = digits_test_pca

gnb = GaussianNB()
gnb.fit(x_train, y_train)
predicted_y_test = gnb.predict(x_test)

output_file("results/gaussian.csv", predicted_y_test)
