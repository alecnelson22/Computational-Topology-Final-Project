import os
import pandas as pd
import numpy as np
import gudhi.representations.vector_methods

inFolder = './space_time_barcode/'
outFolder = './space_time_feature/'
outputFilename = 'full_dataset.npz'

dataset_list_filename = './data/dataset_list.csv'
df = pd.read_csv(dataset_list_filename)

list_of_barcode_lists = []
y_avg = []
y_med = []
for index, row  in df.iterrows():
    filename = row['filename']
    avg_risk = row['avg-risk-score']
    med_risk = row['median-risk-score']
    basename = os.path.basename(filename)
    barcode_filename = basename.replace('.csv', '_space_time_barcode.txt')
    if os.path.exists(inFolder + barcode_filename):
        print(barcode_filename, '...')
        with open(inFolder + barcode_filename, 'r') as f:
            file_lines = f.readlines()
            idx_0 = file_lines.index('persistence intervals in dim 0:\n')
            dim_0_rows = file_lines[idx_0+1:]
            data_rows = [ string.strip(' [)\n').split(',') for string in dim_0_rows if string.strip() != '']
            data_rows = [ [0 if x == '' else float(x) for x in row] for row in data_rows]
            data_rows = np.array(data_rows)
            list_of_barcode_lists.append(data_rows)
            y_avg.append(avg_risk)
            y_med.append(med_risk)


max_value = ((0.6**2) + (0.6**2))** 0.5 # the 0.6 is related to our max threshold. The max distance, is then the diagonal across this 0.6x0.6
vectorizor = gudhi.representations.vector_methods.PersistenceImage
vectorizor = vectorizor(resolution = [10, 10], im_range=[0,max_value,0,max_value])
X = vectorizor.transform(list_of_barcode_lists)
np.savez(outFolder + outputFilename, X = np.array(X), y_avg = np.array(y_avg), y_med = np.array(y_med))