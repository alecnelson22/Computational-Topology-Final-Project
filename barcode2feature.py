import os
import pandas as pd
import numpy as np
import gudhi.representations.vector_methods

inFolder = './takens_barcode/'
outFolder = './persistence_image_feature/'
 
dataset_list_filename = './data/dataset_list.csv'
df = pd.read_csv(dataset_list_filename)

def barcode2feature(string_rows, outputFilename):
    data_rows = [ string.strip(' [)\n').split(',') for string in string_rows if string.strip() != '']
    data_rows = [ [0 if x == '' else float(x) for x in row] for row in data_rows]
    vectorizor = gudhi.representations.vector_methods.PersistenceImage
    # if im_range is included you don't need to fit. I think since, we are using takens embedding
    max_value = ((0.6**2) + (0.6**2))** 0.5 # the 0.6 is related to our max threshold. The max distance, is then the diagonal across this 0.6x0.6
    vectorizor = vectorizor(resolution = [10, 10], im_range=[0,max_value,0,max_value])
    vector = vectorizor.transform([np.array(data_rows)])
    np.savetxt(outputFilename, vector, delimiter=' ')
    return

for filename in df['filename']:

    basename = os.path.basename(filename)
    barcode_filename = basename.replace('.csv', '_takens_barcode.txt')
    if os.path.exists(inFolder + barcode_filename):
        print(barcode_filename, '...')
        outputFilename = outFolder + barcode_filename.replace('_takens_barcode.txt', '_persistance_image_feature.npz')
        if os.path.exists(outputFilename):
            print('Already done. Skipping.')
        with open(inFolder + barcode_filename, 'r') as f:
            file_lines = f.readlines()
            idx_0 = file_lines.index('persistence intervals in dim 0:\n')
            # idx_1 = file_lines.index('persistence intervals in dim 1:\n')
            dim_0_rows = file_lines[idx_0+1:]
            # dim_1_rows = file_lines[idx_1+1:]
            barcode2feature(dim_0_rows, outputFilename)
            # barcode2feature(dim_1_rows, 'blarg')