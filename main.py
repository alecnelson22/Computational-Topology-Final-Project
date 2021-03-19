import pandas as pd
# import numpy as np
# import csv
# import os

DATA_FOLDER = './data/'



def main():
    # Load file
    folder = 'Pedestrian Dynamics Data Archive/bottleneck/'
    filename = '250_q_45_h0.csv'
    data = buildDataStructures(folder, filename)
    trajectories, frames, times, col_keys = data
    printBasicStats(data)

    return

def printBasicStats(data):
    trajectories, frames, times, col_keys = data
    trajectory_ids = trajectories.groups.keys()
    number_of_trajectories = len(trajectory_ids)
    number_of_frames = len(times)
    print('# trajectories: {}'.format(number_of_trajectories))
    print('# frames: {}'.format(number_of_frames))
    return

def basicPlot(data):
    trajectories, frames, times, col_keys = data

    return

def buildDataStructures(folder, filename):
    '''
    Returns tuple with 4 values:
        trajectories - Pandas GroupBy object where each group is a trajectory
        frames - Pandas GroupBy object where each group is a list of points at one point in time
        times - a sorted list of times. Useful as keys for 'frames' since that isn't can't be sorted
        col_keys - tuple with 4 relevant column key strings (time, x, y, id)
    '''
    data_frame = pd.read_csv(DATA_FOLDER + folder + filename)

    # Sanity check columns, get variants of keys
    t_key = checkColumn(['time', 't', 'T', 'Time'], data_frame)
    x_key = checkColumn(['x', 'X'], data_frame)
    y_key = checkColumn(['y', 'Y'], data_frame)
    id_key = checkColumn(['id', 'ID'], data_frame)

    col_keys = (t_key, x_key, y_key, id_key)

    # group points into trajectories based on trajectory id
    trajectories = data_frame.groupby(id_key)
    # sort trajectories based on time
    for id, trajectory in trajectories:
        trajectory.sort_values(t_key)

    # group data by time
    frames = data_frame.groupby(t_key)

    times = list(frames.groups.keys())
    times.sort()
    return trajectories, frames, times, col_keys

def checkColumn(key_options, data_frame):
    for key in key_options:
        if key in data_frame.columns:
            return key
    quit('Failed to find required key (' + ','.join(key_options) + ')')

if __name__ == '__main__':
    main()