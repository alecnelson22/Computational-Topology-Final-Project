import pandas as pd
# import numpy as np
# import csv
# import os

DATA_FOLDER = './data/'

# Load csv data to dict (csv formatted)
# fd = 'data/data/Pedestrian Dynamics Data Archive/hallway/'
# fn = 'BI_CORR_400_A_1.csv'

# data = []
# with open(os.path.join(fd, fn), 'r') as file:
#     for i,line in enumerate(csv.reader(file)):
#         if i == 0:
#             attributes = line
#         else:
#             entry = {}
#             for j,item in enumerate(line):
#                 entry[attributes[j]] = item
#             data.append(entry)

# print('Data loaded')


data_frame = pd.read_csv(DATA_FOLDER + 'Pedestrian Dynamics Data Archive/hallway/' + 'BI_CORR_400_A_1.csv')

print(data_frame.describe())