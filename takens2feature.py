import pandas as pd
import os
import numpy as np

from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt


outFolder = './out/'

# data_array = np.load('./data/test/150_q_56_h0_save.npy')
# print(data_array.shape)
# data_array = np.load('./data/test/150_q_56_h0_savez_compressed.npz')
# print(data_array['arr_0'].shape)
data_array = np.load('./out/traj_UNI_CORR_500_09_savez_compressed.npz')['arr_0']
print(data_array.shape)
# print(data_array[0])
# data_array = data_array.reshape(-1, 2)
# mask = ~np.all(data_array == 0, axis=1)
# data_array = data_array[mask]

# print(data_array.shape)
# print(data_array[0])

np.savetxt("./out/big_dumb_list.csv", data_array, delimiter=' ')

print('running ripser...')
rippedData = ripser(data_array)
dgms = rippedData['dgms']

plot_diagrams(dgms, ax=plt.plot())
plt.show()
# plt.savefig(outFolder + 'plot-barcode-' + outFilename + '.png', pad_inches=0)
plt.cla()


# dataset_list_filename = './data/dataset_list.csv'
# df = pd.read_csv(dataset_list_filename)


# for filename in df['filename']:
#     takens_embedding_filename = filename.replace('.csv', '.npz')
#     if os.path.exists(takens_embedding_filename):
#         print(takens_embedding_filename)
#         data_array = np.load('./data/UNI_CORR_500/traj_UNI_CORR_500_09.npz', allow_pickle=True)
#         print(data_array.shape)