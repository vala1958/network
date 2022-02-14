import numpy as np
import matplotlib.pyplot as plt
from brainplot_selfcontained import load_matrix, plot_area_values
from preprocessing import get_date, subdirectory
from fit_and_viterbi import color_function, export_legend
import warnings


def collect_frames_and_mean_nan(trial_num_states, posterior_probs, time_bins, obs, posterior_probs_threshold,
                                centered):
    """
    function collecting the frames/times steps belonging to one hidden state according to the found state sequence
    with the viterbi algorithm. A threshold for the posterior probability can be given (if 'posterior_probs_threshold'
    is set to 1.0, then only frames are taken if p=1.0, if set to a lower limit, then also frames with p<1.0 are taken.
    The mean of these collected frames belonging to one hidden state is calculated. If empty (non existing) states
    are found the relusting NANs are replace with zeros (rising warnings!!). The number of empty states are counted
    and returned for later use. The limits for plotting are also calculated, centered (0 ist in the middle)
    if centered = True, otherwise (False) .

    :return: list of lists of the indices of the frames belonging to one hidden state (sublist), for all hidden
             states,
             list of of the means for the dF/F vector for all states
             number of "real" exisiting states


    """
    #warnings.simplefilter(ignore)

    state_number_list = [x for x in range(0, trial_num_states)]

    lst_index_all_states = []
    lst_average_obs_per_state = []
    number_empty_states = 0

    for item in state_number_list:

        lst_index = []
        for i in range(time_bins):

            if posterior_probs[i, item] >= posterior_probs_threshold:
                lst_index.append(i)

        lst_index_all_states.append(lst_index)
        lst_average_obs_per_state.append(np.mean(obs[lst_index, :], axis=0))

    # change all nan's into zeros (non existing states produce empty sublists in "lst_index_all_states" with no frame
    # indices and therefor "lst_average_obs_per_state" has sublist filled with NANs producing errors.
    # the next code line changes the NANa to zeros

    lst_average_obs_per_state = np.nan_to_num(lst_average_obs_per_state)

    for i in range(len(lst_index_all_states)):
        if len(lst_index_all_states[i]) == 0:
            number_empty_states += 1
        print('Number of frames in state {}:            '.format(i), len(lst_index_all_states[i]))

    # compute the limits for the plots (look vor vmax, vmin in all states)
    lst_limits_min = []
    lst_limits_max = []

    for item in lst_average_obs_per_state:
        lst_limits_min.append(np.min(item))
        lst_limits_max.append(np.max(item))

    vmin = min(lst_limits_min)
    vmax = max(lst_limits_max)

    print()
    print('Minimum dF/F value, original:           ', vmin)
    print('Minimum dF/F value, original:           ', format(vmax, '21'))

    if centered:
        if abs(vmin) >= abs(vmax):
            vmax = - vmin
        else:
            vmin = - vmax
        print()
        print('Minimum dF/F value, limit in plots:     ', vmin)
        print('Maximum dF/F value, limit in plots:     ', format(vmax, '21'))

    return lst_index_all_states, lst_average_obs_per_state, number_empty_states, vmin, vmax


def collect_frames_and_mean(trial_num_states, posterior_probs, time_bins, obs,
                            posterior_probs_threshold, centered):
    """
    function collecting the frames/times steps belonging to one hidden state according to the found state sequence
    with the viterbi algorithm. A threshold for the posterior probability can be given (if 'posterior_probs_threshold'
    is set to 1.0, then only frames are taken if p=1.0, if set to a lower limit, then also frames with p<1.0 are taken.
    The mean of these collected frames belonging to one hidden state is calculated. The limits for plotting are also
    calculated, centered (0 ist in the middle) if centered = True, otherwise (False) it is original

    :return: list of lists of the indices of the frames belonging to one hidden state (sublist), for all hidden
             states,
             list of of the means for the dF/F vector for all states


    """

    state_number_list = [x for x in range(0, trial_num_states)]

    lst_index_all_states = []
    lst_average_obs_per_state = []

    for item in state_number_list:

        lst_index = []
        for i in range(time_bins):

            if posterior_probs[i, item] >= posterior_probs_threshold:
                lst_index.append(i)

        lst_index_all_states.append(lst_index)
        lst_average_obs_per_state.append(np.mean(obs[lst_index, :], axis=0))

    for i in range(len(lst_index_all_states)):
        print('Number of frames in state {}:            '.format(i), len(lst_index_all_states[i]))

    # compute the limits for the plots (look vor vmax, vmin in all states)
    lst_limits_min = []
    lst_limits_max = []

    for item in lst_average_obs_per_state:
        lst_limits_min.append(np.min(item))
        lst_limits_max.append(np.max(item))

    vmin = min(lst_limits_min)
    vmax = max(lst_limits_max)

    print()
    print('Minimum dF/F value, original:           ', vmin)
    print('Minimum dF/F value, original:           ', format(vmax, '21'))

    if centered:
        if abs(vmin) >= abs(vmax):
            vmax = - vmin
        else:
            vmin = - vmax
    print()
    print('Minimum dF/F value, limit in plots:     ', vmin)
    print('Maximum dF/F value, limit in plots:     ', format(vmax, '21'))

    return lst_index_all_states, lst_average_obs_per_state, vmin, vmax


def generate_legend(trial_num_states):
    """
    function that generates a "linear" scatter plot to produce a legend for the state number for the rain bow colors
    which then is exported.
    :param trial_num_states:
    :return:  (export of a legend)
    """
    #plt.style.use('default')
    legend = []

    y_array = np.zeros(trial_num_states)

    for i in range(trial_num_states):
        y_array[i] = 1.1 * i
        legend_element = 'state {}'.format(i)
        legend.append(legend_element)

    c = color_function(y_array)

    fig = plt.figure(figsize=(1, 2))
    for i in range(trial_num_states):
        plt.scatter(1, y_array[i], s=80, c=c[i], label=legend[i])

    # plt.axis('off')
    legend = plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
    plt.xticks([])
    plt.yticks([])

    # the legend for later use
    export_legend(legend)
    plt.show()
    return fig


def plot_brain_states(trial_num_states,lst_average_obs_per_state, vmin, vmax, time_bins, name, trial_type,
                      delay, save_figures, file, title, run):
    """
    function to plot the brain states. each brainstate is one image and each is saved individually
    """

    allenMap, allenIndices = load_matrix('')
    state_number_list = [x for x in range(0, trial_num_states)]

    for i in range(len(lst_average_obs_per_state)):

        fig, ax = plt.subplots(1, 1, figsize=(3, 3), tight_layout=False)

        plot_area_values(fig, ax, allenMap, allenIndices, lst_average_obs_per_state[i],
                         vmin=vmin,
                         vmax=vmax,
                         haveColorBar=True)
        ax.set_title('activity of the 27 Rois: {}, {}, {}\n{},{}, state {}  '.format(name, trial_type, time_bins,
                                                                                     delay, title,
                                                                                     state_number_list[i]))

        if save_figures:
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/brainmap_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                             get_date(), state_number_list[i], run))

    # plot the as png exported legend
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(4, 2))

    img = plt.imread('legend.png')
    plt.imshow(img)
    plt.axis('off')

    if save_figures:
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/legend.png')

    plt.show()


def plot_brain_states_nan(trial_num_states,lst_average_obs_per_state, vmin, vmax, time_bins,
                          name, trial_type, delay, save_figures, file, title, kind_prepro, run):
    """
    function to plot the brain states. each brainstate is one image and each is saved individually, including the
    possibility of empty (non existing) states
    """

    allenMap, allenIndices = load_matrix('')
    state_number_list = [x for x in range(0, trial_num_states)]

    for i in range(len(lst_average_obs_per_state)):

        sum_mean_all_roi = sum(lst_average_obs_per_state[i])

        if sum_mean_all_roi == 0:
            print('state {} does not exist!'.format(i))

        else:
            fig, ax = plt.subplots(1, 1, figsize=(6,6), tight_layout=False)

            plot_area_values(fig, ax, allenMap, allenIndices, lst_average_obs_per_state[i],
                             vmin=vmin,
                             vmax=vmax,
                             haveColorBar=True)
            ax.set_title('activity of the 27 Rois: {}, {}, {}\n{}s, {}, state {}  '.format(name, trial_type, time_bins,
                                                                                           delay, title,
                                                                                           state_number_list[i]))

            if save_figures:
                new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
                subdirectory(file, new)
                plt.savefig(new + '/brainmap_{}_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                                    get_date(), state_number_list[i],
                                                                                    kind_prepro, run))

    # plot the as png exported legend
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(4, 2))

    img = plt.imread('legend.png')
    plt.imshow(img)
    plt.axis('off')

    if save_figures:
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/legend.png')

    plt.show()


def get_figsize(lst):
    """
    function to define in a plot with varying number of subplots the figsize and number of colons
    :return:  figsize
    """
    dict_figsize = {'1': (3, 3), '2': (7, 3), '3': (11, 3), '4': (14, 3), '5': (18, 3),
                    '6': (22, 3), '7': (25, 3), '8': (28, 3), '16': (28, 6)}

    if len(lst) == 1:
        figsize = dict_figsize['1']
    elif len(lst) == 2:
        figsize = dict_figsize['2']
    elif len(lst) == 3:
        figsize = dict_figsize['3']
    elif len(lst) == 4:
        figsize = dict_figsize['4']
    elif len(lst) == 5:
        figsize = dict_figsize['5']
    elif len(lst) == 6:
        figsize = dict_figsize['6']
    elif len(lst) == 7:
        figsize = dict_figsize['7']
    elif len(lst) == 8:
        figsize = dict_figsize['8']
    elif len(lst) < 16:
        figsize = dict_figsize['16']

    return figsize


def get_figsize_2(lst):
    """
    function to define in a plot with varying number of subplots the figsize and number of colons
    :return:  figsize
    """
    dict_figsize = {'8': (28, 3), '16': (28, 6), '24': (28, 9), '32': (28,12)}

    if len(lst) <= 8:
        figsize_2 = dict_figsize['8']
    elif len(lst) <= 16:
        figsize_2 = dict_figsize['16']
    elif len(lst) <= 24:
        figsize_2 = dict_figsize['24']
    elif len(lst) < 24:
        figsize_2 = dict_figsize['32']

    return figsize_2


def plot_brain_states_1(trial_num_states, lst_average_obs_per_state,  vmin, vmax, time_bins,
                        name, trial_type, save_figures, file, title, run):
    """
    function to plot the brainstates in one row and stored as one image, color bar only at the last map
    """

    allenMap, allenIndices = load_matrix('')
    state_number_list = [x for x in range(0, trial_num_states)]

    if trial_num_states > 8:
        print('number of states is to large (squeezed plot), please use other plot function')
    else:

        figsize = get_figsize(lst_average_obs_per_state)

        fig, ax = plt.subplots(1, trial_num_states, figsize=figsize, tight_layout=False)

        for i in range(len(lst_average_obs_per_state)):
            if i == len(lst_average_obs_per_state)-1:
                plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=True)
            else:
                plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=False)

            ax[i].set_title('state {}  '.format(state_number_list[i]))

        if save_figures:
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/brainmap_1_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                               get_date(), state_number_list[i], run))

        plt.show()


def plot_brain_states_1_nan(trial_num_states, lst_average_obs_per_state, vmin, vmax, time_bins,
                            name, trial_type, save_figures, file, title, kind_prepro,  run):
    """
    function to plot the brain states in one row and stored as one image, including the option of empty states
    """

    allenMap, allenIndices = load_matrix('')
    state_number_list = [x for x in range(0, trial_num_states)]

    if trial_num_states > 8:
        print('number of states is to large (squeezed plot), please use other plot function')
    else:
        figsize = get_figsize(lst_average_obs_per_state)

        fig, ax = plt.subplots(1, trial_num_states, figsize=figsize, tight_layout=False)

        for i in range(len(lst_average_obs_per_state)):

            sum_mean_all_roi = sum(lst_average_obs_per_state[i])

            if sum_mean_all_roi == 0:
                print('state {} does not exist!'.format(i))

            else:

                if i == trial_num_states - 1:
                    plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                     vmin=vmin,
                                     vmax=vmax,
                                     haveColorBar=True)
                else:
                    plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                     vmin=vmin,
                                     vmax=vmax,
                                     haveColorBar=False)

            ax[i].set_title('state {}  '.format(state_number_list[i]))
            ax[i].set_yticks([])
            ax[i].set_xticks([])

        if save_figures:
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/brainmap_1_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                               get_date(), kind_prepro, run))

        plt.show()



def plot_brain_states_2(trial_num_states, lst_average_obs_per_state, vmin, vmax,  time_bins, name, trial_type,
                        save_figures, file, title, kind_prepro, run):
    """
    function with a double loop for more then 8 states , saved as one image.
    """
    allenMap, allenIndices = load_matrix('')
    state_number_list = [x for x in range(0, trial_num_states)]

    figsize_2 = get_figsize_2(lst_average_obs_per_state)
    number_of_rows = int(np.ceil(len(lst_average_obs_per_state)/8))

    fig, ax = plt.subplots(number_of_rows, 8, figsize=figsize_2, tight_layout=False)
    ax = ax.ravel()
    for i in range(len(lst_average_obs_per_state)):
        if len(lst_average_obs_per_state) < 8:
            if i == len(lst_average_obs_per_state)-1:
                plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=True)
            else:
                plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=False)
        else:
            if i == 7 or i == 15 or i == 23:
                plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=True)
            else:
                plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=False)

        ax[i].set_title('state {} '.format(state_number_list[i]))

    if save_figures:
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/brainmap_2_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                          get_date(), kind_prepro, run))

    plt.show()

def plot_brain_states_2_nan(trial_num_states, lst_average_obs_per_state, vmin, vmax,  time_bins, name, trial_type,
                        save_figures, file, title, kind_prepro, run):
    """
    function with a double loop for more then 8 states , saved as one image, including  the possibility of empty states
     """

    allenMap, allenIndices = load_matrix('')
    state_number_list = [x for x in range(0, trial_num_states)]

    figsize_2 = get_figsize_2(lst_average_obs_per_state)
    number_of_rows = int(np.ceil(len(lst_average_obs_per_state)/8))

    fig, ax = plt.subplots(number_of_rows, 8, figsize=figsize_2, tight_layout=False)
    ax = ax.ravel()

    for i in range(len(lst_average_obs_per_state)):

        sum_mean_all_roi = sum(lst_average_obs_per_state[i])

        if sum_mean_all_roi == 0:
            print('state {} does not exist!'.format(i))
        else:

            if len(lst_average_obs_per_state) < 8:
                if i == len(lst_average_obs_per_state)-1:
                    plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                     vmin=vmin,
                                     vmax=vmax,
                                     haveColorBar=True)

                else:
                    plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                     vmin=vmin,
                                     vmax=vmax,
                                     haveColorBar=False)

            else:
                if i == 7 or i == 15 or i == 23:
                    plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                     vmin=vmin,
                                     vmax=vmax,
                                     haveColorBar=True)
                else:
                    plot_area_values(fig, ax[i], allenMap, allenIndices, lst_average_obs_per_state[i],
                                     vmin=vmin,
                                     vmax=vmax,
                                     haveColorBar=False)

        ax[i].set_title('state {} '.format(state_number_list[i]))
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    if save_figures:
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/brainmap_2_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                           get_date(), kind_prepro, run))

    plt.show()


def get_lst_sub_states(lst, state_number):
    """
    :param   lst of the indices for a state
    :return: list of list: sublists hold the indices of the substates
    """

    lst_index = []
    for i in range(0, len(lst) - 1):
        if lst[i + 1] - lst[i] > 1:
            lst_index.append(i)

    lst_sub_states = []
    start = 0
    for index in lst_index:
        sub_state = lst[start:index + 1]
        lst_sub_states.append(sub_state)
        start = index + 1

    lst_end = lst[start:]
    lst_sub_states.append(lst_end)

    print('Sublist holding the indices of the frames belonging to one substate of state {}:'. format(state_number))
    print()
    print(lst_sub_states)
    print()
    print('number of substates of state {}:    '.format(state_number), len(lst_sub_states))

    return lst_sub_states


def plot_brain_substates(trial_num_states, lst_sub_states, obs, vmin, vmax, state_number, time_bins, name, trial_type,
                         save_figures, file, title, kind_prepro, run):

    if len(lst_sub_states) > 8:
        print('number of substates is to large (>8), please use other function or select manually, maximal 8')
    else:

        allenMap, allenIndices = load_matrix('')
        sub_state_number_list = [x for x in range(0, len(lst_sub_states))]

        figsize = get_figsize(lst_sub_states)

        fig, ax = plt.subplots(1, len(lst_sub_states), figsize=figsize, tight_layout=False)

        for i in range(len(lst_sub_states)):
            average_obs_per_sub_state = (np.mean(obs[lst_sub_states[i], :], axis=0))
            if i == len(lst_sub_states)-1:
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=True)
            else:
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=False)

            ax[i].set_title('state {}, substate {} '.format(state_number,sub_state_number_list[i]))

        if save_figures:
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/brainmap_sub_{}_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                                 kind_prepro, get_date(), state_number,
                                                                                 run))

        plt.show()


def plot_brain_substates_1(trial_num_states, lst_sub_states, obs, vmin, vmax, state_number, time_bins, name, trial_type,
                         save_figures, file, title, run):
    """
    function with a double loop for more then 8 states (does not work probably)
    """
    allenMap, allenIndices = load_matrix('')
    sub_state_number_list = [x for x in range(0, len(lst_sub_states))]

    figsize = get_figsize(lst_sub_states)
    number_of_rows = int(np.ceil(len(lst_sub_states)/8))
    print(number_of_rows)

    fig, ax = plt.subplots(number_of_rows, 8, figsize=figsize, tight_layout=False)
    for j in range(number_of_rows):
        for i in range(8):
            average_obs_per_sub_state = (np.mean(obs[lst_sub_states[(j*8) + i], :], axis=0))
            plot_area_values(fig, ax[j, i], allenMap, allenIndices, average_obs_per_sub_state,
                             vmin=vmin,
                             vmax=vmax,
                             haveColorBar=False)
            ax[j, i].set_title('state {}, substate {} '.format(state_number,sub_state_number_list[i]))

    if save_figures:
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/brainmap_sub_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                             get_date(), state_number, run))

    plt.show()


def plot_brain_substates_2(trial_num_states, lst_sub_states, obs, vmin, vmax, state_number, time_bins, name, trial_type,
                         save_figures, file, title, kind_prepro, run):
    """
    function with a double loop for more then 8 states
    """
    allenMap, allenIndices = load_matrix('')
    sub_state_number_list = [x for x in range(0, len(lst_sub_states))]

    figsize = get_figsize_2(lst_sub_states)
    number_of_rows = int(np.ceil(len(lst_sub_states)/8))

    fig, ax = plt.subplots(number_of_rows, 8, figsize=figsize, tight_layout=False)
    ax = ax.ravel()
    for i in range(len(lst_sub_states)):
        if len(lst_sub_states) < 8:
            if i == len(lst_sub_states)-1:
                average_obs_per_sub_state = (np.mean(obs[lst_sub_states[i], :], axis=0))
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=True)
            else:
                average_obs_per_sub_state = (np.mean(obs[lst_sub_states[i], :], axis=0))
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=False)
        else:
            if i == 7 or i == 15 or i == 23:
                average_obs_per_sub_state = (np.mean(obs[lst_sub_states[i], :], axis=0))
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=True)
            else:
                average_obs_per_sub_state = (np.mean(obs[lst_sub_states[i], :], axis=0))
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=False)

        ax[i].set_title('state {}, substate {} '.format(state_number,sub_state_number_list[i]))

    if save_figures:
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/brainmap_sub_{}_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                                kind_prepro, get_date(), state_number,
                                                                                run))

    plt.show()


def get_start_and_end_indices(number_of_substates, state_number, time_bins):
    """
    function to get the start and end indices of the substates (manually)
    :returns  a list of list, each sublist holds the start and end indices of a specific substate
    """

    lst_start_and_end_indices = []

    for i in range(number_of_substates):
        lst_item = []
        int1 = int(input('start {}:'.format(i)))
        lst_item.append(int1)

        while True:
            int2 = int(input('end {}:'.format(i)))
            lst_item.append(int2)
            if lst_item[1] <= lst_item[0] or lst_item[1] > time_bins + 1:
                print('invalid input: second index must be larger then the first and smaller as the number of frames')
                del lst_item[1]
                continue
            else:
                lst_item[1] += 1
                break

        lst_start_and_end_indices.append(lst_item)
    print()
    print('Indices of the start and end frames belonging to {} substates of state {}:'.format(number_of_substates,
                                                                                                   state_number))
    print()
    print(lst_start_and_end_indices)

    return lst_start_and_end_indices


def plot_brain_substates_start_end(trial_num_states, lst_start_and_end_indices, obs, vmin, vmax, state_number,
                                   time_bins, name, trial_type, save_figures, file, title, kind_prepro, run):

    if len(lst_start_and_end_indices) > 8:
        print('number of substates is to large (>8), please use other function or select manually, maximal 8')
    else:

        allenMap, allenIndices = load_matrix('')
        sub_state_number_list = [x for x in range(0, len(lst_start_and_end_indices))]

        figsize = get_figsize(lst_start_and_end_indices)

        fig, ax = plt.subplots(1, len(lst_start_and_end_indices), figsize=figsize, tight_layout=False)

        for i in range(len(lst_start_and_end_indices)):
            average_obs_per_sub_state = (np.mean(obs[lst_start_and_end_indices[i][0]:lst_start_and_end_indices[i][1],
                                                 :], axis=0))
            if i == len(lst_start_and_end_indices)-1:
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=True)
            else:
                plot_area_values(fig, ax[i], allenMap, allenIndices, average_obs_per_sub_state,
                                 vmin=vmin,
                                 vmax=vmax,
                                 haveColorBar=False)

            ax[i].set_title('state {}, substate {} '.format(state_number,sub_state_number_list[i]))

        if save_figures:
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/brainmap_sub_{}_{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, time_bins, title,
                                                                                 get_date(), state_number, kind_prepro,
                                                                                 run))

        plt.show()



def warning_on():
    warnings.simplefilter('default')


