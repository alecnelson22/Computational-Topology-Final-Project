import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import gudhi
from gudhi.hera import wasserstein_distance
import numpy as np
import math
from gtda.time_series import TakensEmbedding

DEFAULT_DATASET = 'data/UNI_CORR_500/traj_UNI_CORR_500_09.csv'
OUT_FOLDER = './out/'
if not os.path.exists(OUT_FOLDER):
    os.mkdir(OUT_FOLDER)

def main(filename):
    build_trajectory_heatmaps('data/dataset_list.csv')
    # build_all_space_time_point_clouds('./data/dataset_list.csv')
    # build_training_data('./data/dataset_list.csv')
    # generate_takens('data/dataset_list.csv')

    # print('building data structures')
    # Load file and build data structure
    # data = build_data_structures(filename)
    # trajectories, frames, col_keys = data

    # do analysis with data structure
    # print_basic_stats(data)

    # plot_trajectories(data, True)

    # ids = frames['id'].unique()

    # build_training_data(filename)

    # input_filename = os.path.basename(filename).split('.')[0]
    # embedded_pts = create_takens_embedding(data)
    # np.savez_compressed(OUT_FOLDER + input_filename + '_savez_compressed', embedded_pts)
    # np.save(input_filename + '_save', embedded_pts)
    # save_takens_embedding(embedded_pts, input_filename)

    # out_folder = './out/'
    # if not os.path.exists(out_folder):
    #     os.mkdir(out_folder)
    # input_filename = os.path.basename(filename)
    # data_with_risk_scores.to_csv(out_folder + 'risk-' + input_filename, index=False)
    return

def print_basic_stats(data):
    trajectories, frames, col_keys = data
    trajectory_ids = trajectories.groups.keys()
    number_of_trajectories = len(trajectory_ids)
    number_of_frames = len(frames)
    print('# trajectories: {}'.format(number_of_trajectories))
    print('# frames: {}'.format(number_of_frames))
    return


def get_spatial_bounds(dataset_list_filename):
    xmin = 10000
    xmax = -10000
    ymin = 10000
    ymax = -10000
    df = pd.read_csv(dataset_list_filename)
    for index, row in df.iterrows():
        filename = row['filename']
        data = build_data_structures(filename)
        _trajectories, frames, col_keys = data
        xmin_i = frames['x'].min().min()
        xmax_i = frames['x'].max().max()
        ymin_i = frames['y'].min().min()
        ymax_i = frames['y'].max().max()
        if xmin_i < xmin:
            xmin = xmin_i
        if xmax_i > xmax:
            xmax = xmax_i
        if ymin_i < ymin:
            ymin = ymin_i
        if ymax_i > ymax:
            ymax = ymax_i
    return math.floor(xmin), math.ceil(xmax), math.floor(ymin), math.ceil(ymax)


def build_trajectory_heatmaps(dataset_list_filename):
    xmin, xmax, ymin, ymax = get_spatial_bounds(dataset_list_filename)
    scale = 10
    nx = (xmax - xmin) * scale
    ny = (ymax - ymin) * scale

    df = pd.read_csv(dataset_list_filename)
    for index, row in df.iterrows():
        filename = row['filename']
        data = build_data_structures(filename)
        _trajectories, frames, col_keys = data
        heatmap = np.zeros((ny, nx))

        for i, frame in frames:
            x_i = frame['x']
            y_i = frame['y']

            for i, x in x_i.items():
                x = int((x_i[i] - xmin) * scale)
                y = int((y_i[i] - ymin) * scale)
                heatmap[y,x] += 1
        heatmap = heatmap.flatten()
        base_fname = os.path.basename(filename).split('.')[0]
        np.savez_compressed(os.path.join(OUT_FOLDER, base_fname + '_heatmap'), heatmap)
        # plt.imshow(heatmap)
        # plt.show()




def plot_trajectories(data, include_center_of_mass):
    trajectories, frames, col_keys = data
    (t_key, x_key, y_key, id_key) = col_keys
    fig, ax = plt.subplots()
    trajectories.plot(ax = ax, x = x_key, y = y_key, legend = False)
    if include_center_of_mass:
        center_of_mass_list = frames.agg('mean')
        center_of_mass_list.plot(ax = ax, x = x_key, y = y_key, linewidth=3, color='black',legend = False)
    plt.show()
    return

def build_training_data(dataset_list_filename, force_all = False):
    df = pd.read_csv(dataset_list_filename)
    total = len(df)
    for index, row in df.iterrows():
        filename = row['filename']
        to_print = '|---- [{} / {}] {} '.format(index+1, total, filename)
        padding = 99 - len(to_print)
        to_print += padding*'-' + '|'
        print(to_print)
        if force_all or row.isnull().values.any():
            avg, median = get_agg_risk_scores(filename)
            df.loc[index, ['avg-risk-score', 'median-risk-score']] = [avg, median]
            df.to_csv(dataset_list_filename, index=False)
    return

def build_all_simple_features(dataset_list_filename):
    df = pd.read_csv(dataset_list_filename)
    total = len(df)
    X = []
    y_avg = []
    y_med = []
    for index, row in df.iterrows():
        filename = row['filename']
        avg_risk = row['avg-risk-score']
        med_risk = row['median-risk-score']
        to_print = '|---- [{} / {}] {} '.format(index+1, total, filename)
        padding = 99 - len(to_print)
        to_print += padding*'-' + '|'
        print(to_print)
        feature = trajectory_to_simple_feature(filename)
        print(feature)
        X.append(feature)
        y_avg.append(avg_risk)
        y_med.append(med_risk)
    np.savez('./simple_feature/full_dataset.npz', X = np.array(X), y_avg = np.array(y_avg), y_med = np.array(y_med))
    return

def trajectory_to_simple_feature(filename):
    data = build_data_structures(filename)
    trajectories, frames, col_keys = data
    (t_key, x_key, y_key, id_key) = col_keys
    
    num_frames = len(frames)
    avg_trajectories = frames[id_key].count().mean()
    num_trajectories = len(trajectories)

    return np.array([num_frames, avg_trajectories, num_trajectories])

def build_all_space_time_point_clouds(dataset_list_filename):
    df = pd.read_csv(dataset_list_filename)
    total = len(df)
    for index, row in df.iterrows():
        filename = row['filename']
        to_print = '|---- [{} / {}] {} '.format(index+1, total, filename)
        padding = 99 - len(to_print)
        to_print += padding*'-' + '|'
        print(to_print)
        trajectory_to_point_cloud_3D(filename)
    return

def trajectory_to_point_cloud_3D(filename, stride=10):
    data = build_data_structures(filename)
    _trajectories, frames, col_keys = data

    (t_key, x_key, y_key, id_key) = col_keys

    times = list(frames.groups)
    sampled_times = times[::stride]
    groups = [frames.get_group(t) for t in sampled_times]
    sampled_frames = pd.concat(groups)
    sampled_frames = sampled_frames[[t_key, x_key, y_key]]
    outFolder = './space_time_point_cloud/'

    sampled_frames.to_csv(outFolder +  os.path.basename(filename), index=False, header=False)
    return

def get_agg_risk_scores(filename, save_all = True):
    data = build_data_structures(filename)

    data_with_risk_scores = calculate_risk_scores(data)
    if save_all:
        input_filename = os.path.basename(filename)
        data_with_risk_scores.to_csv(OUT_FOLDER + 'risk-' + input_filename, index=False)

    # print(data_with_risk_scores)
    traj_risk_scores = data_with_risk_scores[['id', 'total-risk']].groupby('id').agg('mean')
    # print(traj_risk_scores)
    avg_risk = traj_risk_scores['total-risk'].mean()
    median_risk = traj_risk_scores['total-risk'].median()
    return avg_risk, median_risk


# Creates 2-d scatter plot from x,y coordinates
def scatterplot(coords, title):
    ptr = 0
    for i in range(len(dirs)):
        plt.scatter(coords[ptr:ptr+n_pics, 0], coords[ptr:ptr+n_pics, 1])
        ptr += n_pics
    plt.title(title)
    plt.savefig(title)
    plt.show()


def create_takens_embedding(data):
    _trajectories, frames, col_keys = data
    (t_key, x_key, y_key, id_key) = col_keys

    max_t = frames[t_key].mean().max()
    min_t = frames[t_key].mean().min()
    delta_t = (max_t - min_t) / (len(frames) - 1)

    distances = []
    pairwise_idxs = {}

    frame_num = 0
    lf = len(frames)
    c = 0
    print('Rips/Takens...')
    for _time, frame in frames:
        frame_num += 1
        print('Frame: {} / {}'.format(frame_num, len(frames)), end='\r')
        points = frame[[x_key, y_key]].to_numpy()
        rips = gudhi.RipsComplex(points, max_edge_length=0.6)
        simplex_tree = rips.create_simplex_tree(max_dimension=1)

        for indices, distance in simplex_tree.get_filtration():
            if len(indices) != 2:
                continue
            i,j = indices
            id1 = frame.iloc[i][id_key]
            id2 = frame.iloc[j][id_key]
            if id1 == id2:
                continue
            key = get_dual_key(id1, id2)

            if key in pairwise_idxs:
                idx = pairwise_idxs[key]
                distances[idx][frame_num - 1] = distance
            else:
                pairwise_idxs[key] = c
                distances.append(np.zeros(lf))
                distances[c][frame_num - 1] = distance
                c += 1


    new_distances = []
    for pair in distances:
        if np.count_nonzero(pair) > 2:
            new_distances.append(pair)
    distances = np.array(distances)
    new_distances = np.array(new_distances)
    print('distances.shape', distances.shape)
    print('new_distances.shape', new_distances.shape)

    print('distances shape', np.array(distances).shape)

    # distances= np.load('distances_test_1.npz')['arr_0']
    # np.savez_compressed('distances_test', distances)



    print('Computing Takens Embedding...')
    TE = TakensEmbedding(time_delay=10, dimension=2, flatten=True, stride=20)
    embedded_points = np.array(TE.fit_transform(distances))
    print('embedded points shape', embedded_points.shape)

    # Plotting colors to represent different classes of images
    colors = ['black',
              'lightcoral',
              'red',
              'peru',
              'yellow',
              'lawngreen',
              'royalblue',
              'fuchsia']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # toplot = [5, 105, 205, 305, 405, 505, 605, 705]
    # c = 0
    # for i in toplot:
    for i in range(500):
        #ax1.scatter(embedded_points[i, :, 0], embedded_points[i, :, 1], color=colors[c])
        ax1.scatter(embedded_points[i, :, 0], embedded_points[i, :, 1])
        # c += 1
    plt.show()

    return embedded_points

def save_takens_embedding(data_array, input_filename, reshape=True, remove_zeroes=True):
    if reshape:
        data_array = data_array.reshape(-1, 2)
    if remove_zeroes:
        mask = ~np.all(data_array == 0, axis=1)
        data_array = data_array[mask]

    np.savez_compressed(OUT_FOLDER + input_filename + '_savez_compressed', data_array)
    np.save(OUT_FOLDER + input_filename + '_save', data_array)

def calculate_risk_scores(data):
    trajectories, frames, col_keys = data
    traj_ids = trajectories.groups.keys()
    exposure_matrix = init_exposure_matrix(traj_ids)
    (t_key, x_key, y_key, id_key) = col_keys

    max_t = frames[t_key].mean().max()
    min_t = frames[t_key].mean().min()
    delta_t = (max_t - min_t) / (len(frames) - 1)

    updated_frames = pd.DataFrame()
    total_risk_map = dict()
    frame_num = 0
    for _time, frame in frames:
        frame_num += 1
        print('Frame: {} / {}'.format(frame_num, len(frames)), end='\r')
        points = frame[[x_key, y_key]].to_numpy()
        rips = gudhi.RipsComplex(points, max_edge_length=0.6)
        simplex_tree = rips.create_simplex_tree(max_dimension=1)
        for indices, distance in simplex_tree.get_filtration():
            if len(indices) != 2:
                continue
            i,j = indices
            id1 = frame.iloc[i][id_key]
            id2 = frame.iloc[j][id_key]
            if id1 == id2:
                continue
            key = get_dual_key(id1, id2)
            this_exposure = exposure_function(distance, type='weighted') * delta_t
            exposure_matrix[key] += this_exposure
        risk_so_far_map = dict()
        for index, point in frame.iterrows():
            id1 = point[id_key]
            risk_so_far_inverse = 1
            for id2 in traj_ids:
                if id1 == id2:
                    continue
                else:
                    key = get_dual_key(id1, id2)
                    exposure = exposure_matrix[key]
                    risk_so_far_inverse *= 1 - risk_function(exposure)
            risk_so_far = 1 - risk_so_far_inverse
            risk_so_far_map[id1] = risk_so_far
            total_risk_map[id1] = risk_so_far

        frame['risk-so-far'] = frame[id_key].map(risk_so_far_map)
        updated_frames = updated_frames.append(frame)

    updated_frames['total-risk'] = updated_frames[id_key].map(total_risk_map)
    return updated_frames


def risk_function(exposure):
    # todo move parameter r and k to command line arg (with good defaults)
    r = 0.1 # how quickly you will get infected if you are exposed to someone with covid
    k = 0.1 # chance a person has covid
    return k * (1 - (1 / (r * exposure + 1)))


def exposure_function(distance, type='binary'):
    # todo add different function types
    # todo move function type, and function params to command-line args (with good defaults)
    threshold = 0.6
    if distance >= threshold:
        return 0
    else:
        if type == 'binary':
            return 1
        elif type == 'weighted':
            weight = 1 - distance / threshold
            return weight
        else:
            raise Exception('Unrecognized function type')


def init_exposure_matrix(traj_ids):
    exposure_matrix = dict()
    for id1 in traj_ids:
        for id2 in traj_ids:
            if id1 == id2:
                continue
            key = get_dual_key(id1, id2)
            exposure_matrix[key] = 0
    return exposure_matrix


def get_dual_key(id1, id2):
    keyList = sorted([int(id1), int(id2)])
    key = '-'.join([str(x) for x in keyList])
    return key


def generate_takens(filename):
    df = pd.read_csv(filename)
    for f in df['filename']:
        print('Processing ', f, '...')
        main(f)

def build_data_structures(filename, verbose = False):
    '''
    Returns tuple with 4 values:
        trajectories - Pandas GroupBy object where each group is a trajectory
        frames - Pandas GroupBy object where each group is a list of points at one point in time
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
    
    # add columns for storing risk info
    data_frame['risk-so-far'] = 0
    data_frame['total-risk'] = 0

    # group points into trajectories based on trajectory id
    trajectories = data_frame.copy().groupby(id_key)
    # sort trajectories based on time
    for _id, trajectory in trajectories:
        trajectory.sort_values(t_key)

    # group data by time
    frames = data_frame.groupby(t_key)
    if verbose:
        print(frames.describe())

    return trajectories, frames, col_keys


def check_column(key_options, data_frame):
    for key in key_options:
        if key in data_frame.columns:
            return key
    quit('Failed to find required key (' + ','.join(key_options) + ')')


def plot_risk(filename, ids=[]):
    '''
    Plots risk over time as line graph

    ids - optional list of ints of ids to plot
          if left blank, all ids are plotted
    '''
    def _draw_data(df, ids):
        for i in ids:
            rows = df.loc[data_frame['id'] == i]
            risk = rows['risk-so-far'].tolist()
            plt.plot(risk)

    df = pd.read_csv(filename)
    if len(ids) == 0:
        ids = df['id'].unique()
    _draw_data(df, ids)

    plt.xlabel('Time')
    plt.ylabel('Risk')
    plt.title('Covid Exposure risk')
    plt.show()


def risk_heatmap(filename, res=10):
    """
    Plots heatmap of risk
    Gets x,y coordinate for data point, increments pixel value based on risk-so-far param

    TODO Does not take into account a 6 ft radius of risk
         Only increments a single pixel value for a given data point regardless of res param
    """
    # get coordinate bounds
    df = pd.read_csv(filename)
    xmin = df['x'].min()
    xmax = df['x'].max()
    ymin = df['y'].min()
    ymax = df['y'].max()

    # print('xmin', xmin)
    # print('xmax', xmax)
    # print('ymin', ymin)
    # print('ymax', ymax)

    # create img based on coordinate bounds, scale by res param
    nx = math.ceil((xmax - xmin) * res)
    ny = math.ceil((ymax - ymin) * res)
    img = np.zeros((ny, nx))

    # scales data point d from rmin, rmax data range to new tmin, tmax data range
    def _scale(d, rmin, rmax, tmin, tmax):
        scaled = ((d - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        return scaled

    for i, row in df.iterrows():
        x = row['x']
        y = row['y']
        xscaled = math.floor(_scale(x, xmin, xmax, 0, nx)) - 1
        yscaled = math.floor(_scale(y, ymin, ymax, 0, ny)) - 1
        img[yscaled, xscaled] += row['risk-so-far']

    plt.imshow(img)
    plt.show()


# risk_heatmap('out/risk-BI_CORR_400_A_1.csv')
# main('data/Pedestrian Dynamics Data Archive/bottleneck/150_q_56_h0.csv')
# main('data/UNI_CORR_500/traj_UNI_CORR_500_06.csv')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        main(DEFAULT_DATASET)
        exit(0)
        # quit("Missing filename as command-line argument. e.g. 'python main.py ./data/Simple/two_ortho.csv'")
    filename = sys.argv[1]
    if not os.path.exists(filename):
        quit("Filename '{}' does not exist".format(filename))
    main(filename)
    # plot_risk(filename, ids=[1967, 1934, 1871])  # todo provide ids as command line args
