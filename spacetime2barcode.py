import os
import subprocess
import pandas as pd
import numpy as np

inFolder = './space_time_point_cloud/'
outFolder = './space_time_barcode/'

dataset_list_filename = './data/dataset_list.csv'
df = pd.read_csv(dataset_list_filename)
for filename in df['filename']:

    basename = os.path.basename(filename)
    # takens_embedding_filename = basename.replace('.csv', '_takens_point_cloud.txt')
    if os.path.exists(inFolder + basename):
        print(basename, '...')
        outputFilename = outFolder + basename.replace('.csv', '_space_time_barcode.txt')
        if os.path.exists(outputFilename):
            print('Already done. Skipping.')
            continue
        command_list = ['./ripser', '--format', 'point-cloud', '--dim', '0', inFolder + basename]
        command_string = ' '.join(command_list)
        # print(command_string)
        output = subprocess.check_output(command_list)
        with open(outputFilename, 'wb') as f:
            f.write(output)