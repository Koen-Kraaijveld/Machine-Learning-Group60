import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split


digits = pd.read_csv('data/train.csv')
print(digits.describe(include='all'))

# images = digits.iloc[0:5000,1:]
# labels = digits.iloc[0:5000,:1]
# train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
#
# i=3
# img=images.iloc[i].to_numpy()
# img=img.reshape((28,28))
# plt.imshow(img,cmap='gray')
# plt.title(labels.iloc[i,0])
# # plt.hist(train_images.iloc[i])
# # plt.show()
