import os
import pandas as pd
import numpy as np

inFolder = './persistence_image_feature/'
outFolder = './persistence_image_feature/'
outputFilename = 'full_dataset.npz' 

dataset_list_filename = './data/dataset_list.csv'
df = pd.read_csv(dataset_list_filename)

X = []
y_avg = []
y_med = []
for index, row  in df.iterrows():
    filename = row['filename']
    avg_risk = row['avg-risk-score']
    med_risk = row['median-risk-score']
    print(filename)
    print(avg_risk)
    print(med_risk)

    basename = os.path.basename(filename)
    feature_filename = basename.replace('.csv', '_persistance_image_feature.txt')
    if os.path.exists(inFolder + feature_filename):
        print(feature_filename, '...')
        feature_array = np.loadtxt(inFolder + feature_filename, delimiter = ' ')
        X.append(feature_array)
        y_avg.append(avg_risk)
        y_med.append(med_risk)

np.savez(outFolder + outputFilename, X = np.array(X), y_avg = np.array(y_avg), y_med = np.array(y_med))