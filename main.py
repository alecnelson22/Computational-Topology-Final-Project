import pandas as pd
# import numpy as np
# import csv
# import os

DATA_FOLDER = './data/'



def main():
    # Load file
    folder = 'Pedestrian Dynamics Data Archive/bottleneck/'
    filename = '250_q_45_h0.csv'
    trajectories, frames, times = buildDataStructures(folder, filename)
    

    return

def buildDataStructures(folder, filename):
    data_frame = pd.read_csv(DATA_FOLDER + folder + filename)

    # Sanity check columns, get variants of keys
    tKey = checkColumn(['time', 't', 'T', 'Time'], data_frame)
    xKey = checkColumn(['x', 'X'], data_frame)
    yKey = checkColumn(['y', 'Y'], data_frame)
    idKey = checkColumn(['id', 'ID'], data_frame)

    # group points into trajectories based on trajectory id
    trajectories = data_frame.groupby(idKey)
    # sort trajectories based on time
    for id, trajectory in trajectories:
        trajectory.sort_values(tKey)

    # group data by time
    frames = data_frame.groupby(tKey)

    times = frames.groups.keys()
    times.sort()
    return trajectories, frames, times

def checkColumn(keyOptions, data_frame):
    for key in keyOptions:
        if key in data_frame.columns:
            return key
    # print('Failed to find required key (' + keyOptions.join(',') + ')')
    quit('Failed to find required key (' + ','.join(keyOptions) + ')')

if __name__ == '__main__':
    main()