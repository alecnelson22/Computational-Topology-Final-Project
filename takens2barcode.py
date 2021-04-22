import os
import subprocess
import pandas as pd
import numpy as np

inFolder = './takens_point_cloud/'
outFolder = './takens_barcode/'

dataset_list_filename = './data/dataset_list.csv'
df = pd.read_csv(dataset_list_filename)
for filename in df['filename']:

    basename = os.path.basename(filename)
    takens_embedding_filename = basename.replace('.csv', '_takens_point_cloud.txt')
    if os.path.exists(inFolder + takens_embedding_filename):
        print(takens_embedding_filename, '...')
        outputFilename = outFolder + takens_embedding_filename.replace('_takens_point_cloud.txt', '_takens_barcode.txt')
        if os.path.exists(outputFilename):
            print('Already done. Skipping.')
        command_list = ['./ripser', '--format', 'point-cloud', inFolder + takens_embedding_filename]
        command_string = ' '.join(command_list)
        # print(command_string)
        output = subprocess.check_output(command_list)
        with open(outputFilename, 'wb') as f:
            f.write(output)