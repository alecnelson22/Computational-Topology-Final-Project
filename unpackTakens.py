import os
import pandas as pd
import numpy as np

inFolder = './out/'
outFolder = './takens_point_cloud/'

dataset_list_filename = './data/dataset_list.csv'
df = pd.read_csv(dataset_list_filename)
for filename in df['filename']:

    basename = os.path.basename(filename)
    takens_embedding_filename = basename.replace('.csv', '_savez_compressed.npz')
    if os.path.exists(inFolder + takens_embedding_filename):
        print(takens_embedding_filename, '...')
        data_array = np.load(inFolder + takens_embedding_filename)['arr_0']
        print('Shape: ', data_array.shape)
        outputFilename = takens_embedding_filename.replace('_savez_compressed.npz', '_takens_point_cloud.txt')
        np.savetxt(outFolder + outputFilename, data_array, delimiter = ' ')