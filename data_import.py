import numpy as np
import h5py
import pandas as pd



def metadata_to_nparray(session):
    df = pd.read_csv(session)
    trialType = df['trialType'].to_numpy()

    return trialType




def stacking(a1,a2):
    return np.vstack((a1,a2))

def stackingh(a1,a2):
    return np.hstack((a1,a2))


def get_data(lst, path, label, trial_type):

    """function reading the data form the H5py File. The path to that file has to be specified in the
    variable "path" in the notebook. It scrolls through the lst (also to be defined in the notebook) containing
    all the sessions which are studied. Only the in the notebook specified trial types are taken. Option of taking
    all trials (unsorted not included). In the notebook currently get_data_1 active, see below.

    lst:         list of the sessions to be taken
    path:        string to define the path to the H5py.File
    label:       mouse_x_ (string), to build the correct csv file name in which the metadata (trial typ) is stored
    trial_type:  string, to define the trial type

    returns:      the trial type sorted data of the sessions defined in the lst """


    lst_data = []

    for item in lst:

        mouse1 = h5py.File(path)

        # Get the HDF5 group
        group = mouse1['bn_trial']
        data_session = group[item][()]

        mouse1.close()

        # define function for reading/take only trialType/to np array
        m_meta = metadata_to_nparray(label + str(item))

        # take only the "rows" which correspond to a trial type (e.g. hits, fancy indexing)
        data = data_session[m_meta == trial_type]

        lst_data.append(data)

    data_sorted = np.concatenate((lst_data), axis=0)

    return data_sorted


def nice_print(data, trial_type):
    # nice print:
    s1 = 'Number of {} Trials:'.format(trial_type)
    s2 = data.shape[0]

    print('shape/dimensions of the data array:  ', np.shape(data))
    print('{0:38}{1}'.format(s1, s2))
    print('Number of time steps (frames):       ', data.shape[1])
    print('Number of dimensions:                ', data.shape[2])


def get_data_1(lst, path, label, trial_type):

    """function reading the data form the H5py File. The path to that file has to be specified in the
    variable "path" in the notebook. It scrolls through the lst (also to be defined in the notebook) containing
    all the sessions which are studied. Only the in the notebook specified trial types are taken. Here the option 'All'
    is included, to allow the fitting of the model on all unsorted data. Currently activated in the notebook

    lst:         list of the sessions to be taken
    path:        string to define the path to the H5py.File
    label:       mouse_x_ (string), to build the correct csv file name in which the metadata (trial typ) is stored
    trial_type:  string, to define the trial type

    returns:      the trial type sorted data of the sessions defined in the lst """

    lst_data = []

    for item in lst:

        lst_trial_type = ['Hit', 'CR', 'FA', 'Early', 'Miss']

        mouse1 = h5py.File(path)

        # Get the HDF5 group
        group = mouse1['bn_trial']
        data_session = group[item][()]

        mouse1.close()

        # define function for reading/take only trialType/to np array
        m_meta = metadata_to_nparray(label + str(item))

        if trial_type in lst_trial_type:
            # take only the "rows" which correspond to a trial type (e.g. hits, fancy indexing)
            data = data_session[m_meta == trial_type]

        else:
            data = data_session

        lst_data.append(data)

    data_sorted = np.concatenate(lst_data, axis=0)

    return data_sorted


def lst_obs_3d_to_lst_obs_2d(lst):
    """
    function to generate from the list holding 3dim data as elements (blocks of sessions) a list of single trials (2d arrays) as elements.
    allows to work with sessions with different number of frames (get_data_2).
    :param lst:
    :return:
    """
    lst_obs = []
    for i in range(len(lst)):
        for j in range(lst[i].shape[0]):
            lst_obs.append(lst[i][j])

    return lst_obs


def get_data_2(lst, path, label, trial_type):

    """function reading the data form the H5py File. The path to that file has to be specified in the
    variable "path" in the notebook. It scrolls through the lst (also to be defined in the notebook) containing
    all the sessions which are studied. Only the in the notebook specified trial types are taken. Here the option 'All'
    is included, to allow the fitting of the model on all unsorted data. Currently activated in the notebook. This
    function allows to take sessions with different numbers of frames. But it returns a list not a 3d array, and
    therefore no averages can be calculated.

    lst:         list of the sessions to be taken
    path:        string to define the path to the H5py.File
    label:       mouse_x_ (string), to build the correct csv file name in which the metadata (trial typ) is stored
    trial_type:  string, to define the trial type

    returns:      the trial type sorted data of the sessions defined in the lst """

    lst_obs_3d = []

    for item in lst:

        lst_trial_type = ['Hit', 'CR', 'FA', 'Early', 'Miss']

        mouse1 = h5py.File(path)

        # Get the HDF5 group
        group = mouse1['bn_trial']
        data_session = group[item][()]

        mouse1.close()

        # define function for reading/take only trialType/to np array
        m_meta = metadata_to_nparray(label + str(item))

        if trial_type in lst_trial_type:
            # take only the "rows" which correspond to a trial type (e.g. hits, fancy indexing)
            data = data_session[m_meta == trial_type]

        else:
            data = data_session

        lst_obs_3d.append(data)

    lst_obs = lst_obs_3d_to_lst_obs_2d(lst_obs_3d)

    return lst_obs


def nice_print_2(lst_obs, trial_type):
    # nice print:
    s1 = 'Number of {} Trials:'.format(trial_type)
    s2 = len(lst_obs)
    s3 = lst_obs[0].shape[1]
    print('{0:38}{1}'.format(s1, s2))
    print('Number of dimensions:                ', s3)



def get_data_3(lst_obs_all, lst, path, label, trial_type):

    """function reading the data form the H5py File. The path to that file has to be specified in the
    variable "path" in the notebook. It scrolls through the lst (also to be defined in the notebook) containing
    all the sessions which are studied. Only the in the notebook specified trial types are taken. Here the option 'All'
    is included, to allow the fitting of the model on all unsorted data. This function allows to take sessions with
    different numbers of frames and different mice.  But it returns a list not a 3d array, and
    therefore no averages can be calculated. The lst_sessions for different mice has to be added in "a loop",
    meaning each mouse has to be selected in block 1b, their sessions have to be defined in block 3b. The lst_obs_all
    holds the 3d arrays for all the selected mice (appended in each "loop").

    lst_obs_all: list of 3d arrays for the selected mouse, allows to collect data for more then one mouse.
    lst:         list of the sessions to be taken
    path:        string to define the path to the H5py.File
    label:       mouse_x_ (string), to build the correct csv file name in which the metadata (trial typ) is stored
    trial_type:  string, to define the trial type

    returns:      the trial type sorted data of the sessions defined in the lst """

    lst_obs_3d = []
    lst_obs = []

    for item in lst:

        lst_trial_type = ['Hit', 'CR', 'FA', 'Early', 'Miss']

        mouse1 = h5py.File(path)

        # Get the HDF5 group
        group = mouse1['bn_trial']
        data_session = group[item][()]

        mouse1.close()

        # define function for reading/take only trialType/to np array
        m_meta = metadata_to_nparray(label + str(item))

        if trial_type in lst_trial_type:
            # take only the "rows" which correspond to a trial type (e.g. hits, fancy indexing)
            data = data_session[m_meta == trial_type]

        else:
            data = data_session

        lst_obs_3d.append(data)

    lst_obs_all.append(lst_obs_3d)

    for item in lst_obs_all:
        lst = lst_obs_3d_to_lst_obs_2d(item)
        lst_obs = lst_obs + lst

        print('Length of the list "lst_obs":', len(lst_obs))

    return lst_obs
