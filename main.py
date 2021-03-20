import sys, os
import pandas as pd
import matplotlib.pyplot as plt

def main(filename):
    # Load file and build data structure
    data = build_data_structures(filename)
    trajectories, frames, times, col_keys = data

    # do analysis with data structure
    print_basic_stats(data) 
    plot_trajectories(data, True)
    return

def print_basic_stats(data):
    trajectories, frames, times, col_keys = data
    trajectory_ids = trajectories.groups.keys()
    number_of_trajectories = len(trajectory_ids)
    number_of_frames = len(times)
    print('# trajectories: {}'.format(number_of_trajectories))
    print('# frames: {}'.format(number_of_frames))
    return

def plot_trajectories(data, include_center_of_mass):
    trajectories, frames, times, col_keys = data
    (t_key, x_key, y_key, id_key) = col_keys
    fig, ax = plt.subplots()
    trajectories.plot(ax = ax, x = x_key, y = y_key, legend = False)
    if include_center_of_mass:
        center_of_mass_list = frames.agg('mean')
        center_of_mass_list.plot(ax = ax, x = x_key, y = y_key, linewidth=3, color='black',legend = False)
    plt.show()
    return

def build_data_structures(filename, verbose = False):
    '''
    Returns tuple with 4 values:
        trajectories - Pandas GroupBy object where each group is a trajectory
        frames - Pandas GroupBy object where each group is a list of points at one point in time
        times - a sorted list of times. Useful as keys for 'frames' since that isn't can't be sorted
        col_keys - tuple with 4 relevant column key strings (time, x, y, id)
    '''
    data_frame = pd.read_csv(filename)
    if verbose:
        print(data_frame.describe())

    # Sanity check columns, get variants of keys
    t_key = check_column(['time', 't', 'T', 'Time'], data_frame)
    x_key = check_column(['x', 'X'], data_frame)
    y_key = check_column(['y', 'Y'], data_frame)
    id_key = check_column(['id', 'ID'], data_frame)
    col_keys = (t_key, x_key, y_key, id_key)

    # Filter to only the relevant 4 columns to remove junk and make any later calculations faster
    data_frame = data_frame[list(col_keys)]

    # group points into trajectories based on trajectory id
    trajectories = data_frame.copy().groupby(id_key)
    # sort trajectories based on time
    for _id, trajectory in trajectories:
        trajectory.sort_values(t_key)

    # group data by time
    frames = data_frame.groupby(t_key)
    if verbose:
        print(frames.describe())

    times = list(frames.groups.keys())
    times.sort()
    return trajectories, frames, times, col_keys

def check_column(key_options, data_frame):
    for key in key_options:
        if key in data_frame.columns:
            return key
    quit('Failed to find required key (' + ','.join(key_options) + ')')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        quit("Missing filename as command-line argument. e.g. 'python main.py ./data/Simple/two_ortho.csv'")
    filename = sys.argv[1]
    if not os.path.exists(filename):
        quit("Filename '{}' does not exist".format(filename))
    main(filename)