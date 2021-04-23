import os
import numpy as np

inFolder = './persistence_image_feature/'
outFolder = './persistence_image_feature/'

inFilename = 'full_dataset.npz' 


full_dataset = np.load(inFolder + inFilename)

X = full_dataset['X']
y_avg = full_dataset['y_avg']
y_med = full_dataset['y_med']

print(X.shape)
print(y_avg.shape)
print(y_med.shape)

print(X[0])
print(y_avg[0])
print(y_med[0])

# TODO train regressor on X vs y