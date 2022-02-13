import numpy as np
import ssm
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from ssm.util import one_hot, find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap
from preprocessing import get_date, subdirectory


def color_function(lst):
    evenly_spaced_interval = np.linspace(0, 1, len(lst))
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    return colors


def get_params_for_fitting():

    lst_model = ['gaussian', 'diagonal_gaussian','studentst', 't', 'diagonal_t', 'diagonal_studentst', 'exponential',
                 'bernoulli', 'categorical', 'input_driven_obs', 'poisson','vonmises', 'ar', 'autoregressive',
                 'no_input_ar', 'diagonal_ar', 'diagonal_autoregressive', 'independent_ar', 'robust_ar',
                 'no_input_robust_ar', 'robust_autoregressive', 'diagonal_robust_ar', 'diagonal_robust_autoregressive']

    lst_optimizer = ['em', 'sgd']

    trial_num_states = input("Number of hidden states:   ")
    trial_num_states = int(trial_num_states)

    N_iters = input('number of iterations:      ')
    N_iters = int(N_iters)

    while True:
        model = input('Emission model:            ')
        if model not in lst_model:
            print('invalid input, check spelling')
            continue
        else:
            break

    while True:
        optimizer = input('Optimizer:                 ')
        if optimizer not in lst_optimizer:
            print('invalid input, check spelling')
            continue
        else:
            break

    return trial_num_states, N_iters, model, optimizer,


def get_kind_of_obs(obs_aver, obs_single, obs_conc, obs_transformed_FA, obs_transformed_FA_s, obs_transformed_FA_c,
                    obs_transformed_PCA, obs_transformed_PCA_s, obs_transformed_PCA_c):

    """ this function is not used in the notebook, in the workflow it is not convenietn"""

    lst_kinds= ['obs_aver', 'obs_conc', 'obs_transformed_FA', 'obs_transformed_FA_c', 'lst_obs_transformed_FA',
                'obs_transformed_PCA', 'obs_transformed_PCA_c']

    lst_kinds_vit = ['obs_aver', 'obs_single', 'obs_transformed_FA', 'obs_transformed_FA_s',
                     'obs_transformed_PCA', 'obs_transformed_PCA_s']
    while True:
        kind_obs_fit = input('kind of observations for fitting algorithm:      ')
        if kind_obs_fit not in lst_kinds:
            print('invalid input, check spelling or list above, or makes no sense')
            continue
        else:
            break

    while True:
        kind_obs_vit = input('kind of observations for Viterbi algorithm:      ')
        if kind_obs_vit not in lst_kinds_vit:
            print('invalid input, check spelling or list above, or makes not sense')
            continue
        else:
            break

    if kind_obs_fit == 'obs_aver':
        obs_fit = obs_aver
        type_fit = 'fit: aver_org'

    elif kind_obs_fit == 'obs_conc':
        obs_fit = obs_conc
        type_fit = 'fit: conc_org'

    elif kind_obs_fit == 'obs_transformed_FA':
        obs_fit = obs_transformed_FA
        type_fit = 'fit: aver_FA'

    elif kind_obs_fit == 'obs_transformed_FA_c':
        obs_fit = obs_transformed_FA_c
        type_fit = 'fit: conc_FA'

    elif kind_obs_fit == 'obs_transformed_PCA':
        obs_fit = obs_transformed_PCA
        type_fit = 'fit: aver_PCA'

    elif kind_obs_fit == 'obs_transformed_PCA_':
        obs_fit = obs_transformed_PCA_c
        type_fit = 'fit: conc_PCA'

    if kind_obs_vit == 'obs_aver':
        obs_vit = obs_aver
        title = 'averaged'
        kind_prepro = 'nil'

    elif kind_obs_vit == 'obs_single':
        obs_vit = obs_single
        title = 'averaged'
        kind_prepro = 'nil'

    elif kind_obs_vit == 'obs_transformed_FA':
        obs_vit = obs_transformed_FA
        title = 'averaged'
        kind_prepro = 'FA'

    elif kind_obs_vit == 'obs_transformed_FA_s':
        obs_vit = obs_transformed_FA_s
        title = 'single'
        kind_prepro = 'FA'

    elif kind_obs_vit == 'obs_transformed_PCA':
        obs_vit = obs_transformed_PCA
        title = 'averaged'
        kind_prepro = 'PCA'

    elif kind_obs_vit == 'obs_transformed_PCA_s':
        obs_vit = obs_transformed_PCA_s
        title = 'single'
        kind_prepro = 'PCA'

    return obs_fit, obs_vit, title, kind_prepro, type_fit


def get_kind_of_observations_fit1(obs_aver):
    obs_fit = obs_aver
    type_fit = 'fit: aver'

    return obs_fit, type_fit


def get_kind_of_observations_fit2(obs_conc):
    obs_fit = obs_conc
    type_fit = 'fit: conc'

    return obs_fit, type_fit


def get_kind_of_observations_fit3(obs_transformed_FA):
    obs_fit = obs_transformed_FA
    type_fit = 'fit: aver_FA'

    return obs_fit, type_fit


def get_kind_of_observations_fit4(obs_transformed_FA_c):
    obs_fit = obs_transformed_FA_c
    type_fit = 'fit: conc_FA'

    return obs_fit, type_fit


def get_kind_of_observations_fit5(lst_obs_transformed_FA):
    obs_fit = lst_obs_transformed_FA
    type_fit = 'fit: lst_FA'

    return obs_fit, type_fit


def get_kind_of_observations_fit6(obs_transformed_PCA):
    obs_fit = obs_transformed_PCA
    type_fit = 'fit: aver_PCA'

    return obs_fit, type_fit


def get_kind_of_observations_fit7(obs_transformed_PCA_c):
    obs_fit = obs_transformed_PCA_c
    type_fit = 'fit: conc_PCA'

    return obs_fit, type_fit


def get_kind_of_observations_vit1(obs_aver):
    obs_vit = obs_aver
    title = 'averaged'
    kind_prepro = 'nil'

    return obs_vit, title, kind_prepro


def get_kind_of_observations_vit2(obs_single, trial_index):
    obs_vit = obs_single
    title = 'single, index {}'.format(trial_index)
    kind_prepro = 'nil'

    return obs_vit, title, kind_prepro


def get_kind_of_observations_vit3(obs_transformed_FA):
    obs_vit = obs_transformed_FA
    title = 'averaged'
    kind_prepro = 'FA'

    return obs_vit, title, kind_prepro


def get_kind_of_observations_vit4(obs_transformed_FA_s, trial_index):
    obs_vit = obs_transformed_FA_s
    title = 'single, index {}'.format(trial_index)
    kind_prepro = 'FA'

    return obs_vit, title, kind_prepro


def get_kind_of_observations_vit5(obs_transformed_PCA):
    obs_vit = obs_transformed_PCA
    title = 'averaged'
    kind_prepro = 'PCA'

    return obs_vit, title, kind_prepro


def get_kind_of_observations_vit6(obs_transformed_PCA_s, trial_index):
    obs_vit = obs_transformed_PCA_s
    title = 'single, index {}'.format(trial_index)
    kind_prepro = 'PCA'

    return obs_vit, title, kind_prepro


def print_to_check(name, trial_type, frames, delay):

    lst_check1 = ['name      ', 'trial_type', 'frames    ', 'delay     ']
    lst_check2 = [name, trial_type, frames, delay]
    print()
    for item1, item2  in zip(lst_check1, lst_check2):
        print(item1 +':        ', item2)


def print_to_check_1(obs_vit, lst_params):

    print('time_bins:            ', np.shape(obs_vit)[0])       # time_bins: number of time bins (time steps or frames)
    print('obs_dim:              ', np.shape(obs_vit)[1])       # obs-dim: data dimension:
    print('trial_num_states:     ', lst_params[0])              # trial_num_states: number of hidden states
    print('N_iters:              ', lst_params[1])              # N_iters: number of iterations
    print('model:                ', lst_params[2])              # model: the selected emission model
    print('optimizer:            ', lst_params[3])              # optimizer: the selected optimizer


def print_to_check_2(obs_vit, lst_params, type_fit, title, kind_prepro):

    print('time_bins:            ', np.shape(obs_vit)[0])       # time_bins: number of time bins (time steps or frames)
    print('obs_dim:              ', np.shape(obs_vit)[1])       # obs-dim: data dimension:
    print('trial_num_states:     ', lst_params[0])              # trial_num_states: number of hidden states
    print('N_iters:              ', lst_params[1])              # N_iters: number of iterations
    print('model:                ', lst_params[2])              # model: the selected emission model
    print('optimizer:            ', lst_params[3])              # optimizer: the selected optimizer
    print('data for fit:         ', type_fit)                   # which data is taken for the fit input
    print('data for viterbi      ', title, kind_prepro)         # which data is taken for viterbi algorithm input


def fitting(trial_num_states, obs_dim, model, obs, optimizer, N_iters):
    """
    fits the model parameter (transition matrix/emission matrix) to the data and returns the likelehoods for
    each iteration
    :param trial_num_states:   integer, number of hidden states
    :param obs_dim:            integer, dimension of the input channels (original 27, reduced: 2-4)
    :param model:              string, distribution model of the data (eg. gaussian)
    :param obs:                2d array, (time_bins, obs_dim), data
    :param optimizer:          string ('em' or 'sgd'
    :param N_iters:            integer, number of iterations
    :return:                   floats, hmm_lls for each iteraton
    """

    hmm = ssm.HMM(trial_num_states, obs_dim, observations= model)

    hmm_lls = hmm.fit(obs, method=optimizer, num_iters=N_iters, init_method="kmeans")

    return hmm, hmm_lls


def plot_fitting_curve(hmm_lls, optimizer, save_figures, file, name, trial_type, time_bins,
                       trial_num_states, title, kind_prepro, run):

    fig = plt.figure(figsize=(8,6))

    plt.plot(hmm_lls, label="EM")

    #if true_ll is known:
    # plt.plot([0, N_iters], true_ll * np.ones(2), ':k', label="True")

    plt.xlabel('{} Iteration'.format(optimizer))
    plt.ylabel("Log Probability")
    plt.legend(loc="lower right")

    if save_figures:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/fit_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, kind_prepro, get_date(),
                                                                 run, bbox_inches='tight',))
    return fig


def state_sequence(hmm, obs):
    """
    function to calculate the most probable state sequence using a fitted model. (calls source code)
    :param obs: data (2d array or list)
    :return:    most probable state sequence
    """

    viterbi_states = hmm.most_likely_states(obs)
    print('most probable state sequence:')
    print(viterbi_states)

    return viterbi_states


def plot_state_sequence(viterbi_states, trial_num_states, time_bins, name, trial_type, delay, title, kind_prepro,
                        type_fit, save_figures, file, run):
    """

    :param viterbi_states:    most probable state sequence (list)
    :param trial_num_states   integer, number of hidden states (to produce a list of integers up to that number)
                              (trial_num_states)
    :return:                  figure of the state sequence
    """
    trial_lst_states = [i + 1 for i in range(0, trial_num_states)]

    plt.rcParams["axes.grid"] = False
    cmap_trial = gradient_cmap(color_function(trial_lst_states))

    fig = plt.figure(figsize=(8, 2))
    plt.imshow(viterbi_states[None, :], aspect="auto", cmap=cmap_trial,
               vmin=0, vmax=len(color_function(trial_lst_states)) - 1,
               extent=(0, time_bins, 0.8, 0.82),
               alpha=1)
    plt.xlim(0, time_bins)
    plt.ylabel("Viterbi ")
    plt.yticks([])
    plt.xlabel("frames")
    plt.title('state sequence: {}, {}, {}, {}, {}s, {}, {}, {}  '.format(name, trial_type, time_bins, trial_num_states,
                                                                        delay, title, kind_prepro, type_fit))
    plt.tight_layout()

    if save_figures:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/sequence {}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, kind_prepro,
                                                                         get_date(), run, bbox_inches="tight"))

    return fig


def state_occupancy_time(viterbi_states,trial_num_states):
    """
    find state occupancy
    code: example [0,0,1,1,1,2,3,3,] --> [0,1,2,3], [2,3,1,2])
    Viterbi: nicht permuted, inferred: viterbi permuted

    :return:
    """

    viterbi_state_list, viterbi_durations = ssm.util.rle(viterbi_states)

    #print(viterbi_state_list)
    #print(viterbi_durations)

    # Rearrange the lists of durations to be a nested list where
    # the nth inner list is a list of durations for state n, sortiert die durations  nach states
    vit_durs_stacked = []

    for s in range(trial_num_states):
        vit_durs_stacked.append(viterbi_durations[viterbi_state_list == s])

    vit_durs_stacked = np.array(vit_durs_stacked, dtype="object")

    # calculate sum in each element of stack list to get to total occupancy time per state
    lst_viterbi_total_occupancy_time = []

    for item in vit_durs_stacked:
        lst_viterbi_total_occupancy_time.append(sum(item))


    print('occupancy time of hidden states: ',  lst_viterbi_total_occupancy_time)

    return vit_durs_stacked, lst_viterbi_total_occupancy_time


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax[1].text(rect.get_x() + rect.get_width() / 2., 1. * height,
                   '%d' % int(height),
                   ha='center', va='bottom')

    return autolabel


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_occupation_duration(trial_num_states, vit_durs_stacked,  name, trial_type, time_bins,
                             delay, title, kind_prepro, type_fit, lst_viterbi_total_occupancy_time,
                             save_figures, file, run):
    """

    :return:
    """
    trial_lst_states = [i + 1 for i in range(0, trial_num_states)]
    # color function zur defintion der colors für  trial und true number of states
    colors_trial = color_function(trial_lst_states)

    # x Achsen defintion
    x_trial = []
    for i in range(len(trial_lst_states)):
        x_trial.append('state {}'.format(i))

    # trial states viterbi (not permuted)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    # plt.style.use('seaborn')
    ax[0].hist(vit_durs_stacked, label=['state ' + str(s) for s in range(trial_num_states)],
               color=colors_trial, alpha=1)
    ax[0].set_xlabel('Duration')
    ax[0].set_ylabel('Frequency')
    ax[0].legend()
    ax[0].set_title(
        'Histogram State Durations: {}, {}, {}, {}, {}s, {}, {}\n {}  '.format(name, trial_type, time_bins,
                                                                               trial_num_states, delay, title,
                                                                               kind_prepro, type_fit))

    rect1 = ax[1].bar(x_trial, lst_viterbi_total_occupancy_time, width=0.4, label=x_trial, color=colors_trial,
                      alpha=1)
    ax[1].set_ylabel('total occupancy time')
    ax[1].set_title(
        'occupancy time: {}, {}, {}, {}, {}s, {}, {}\n {}  '.format(name, trial_type, time_bins, trial_num_states,
                                                                    delay, title, kind_prepro, type_fit))
    autolabel(rect1, ax)

    if save_figures:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/occupancy {}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, kind_prepro,
                                                                          get_date(), run,
                                                                          bbox_inches="tight"))

       # the lengend for later use, new: there is an extra function "generate_legend" in map_to_brain
       #export_legend(legend)

    return fig



def plot_obs_with_state_sequence(trial_num_states, obs, viterbi_states, time_bins, name, trial_type, delay, title,
                                 save_figures, file, kind, run):
    """
    plot the obs before FA, underlayed with viterbi sequence color plot
    :return:   fig
    """

    trial_lst_states = [i + 1 for i in range(0, trial_num_states)]
    cmap_trial = gradient_cmap(color_function(trial_lst_states))

    lim = 1.05 * abs(obs).max()

    if kind == 0:
        fig = plt.figure(figsize=(10, 10))
    else:
        fig = plt.figure(figsize=(10, 5))

    for d in range(obs.shape[1]):
        plt.plot(obs[:, d] + lim * d, '-k')

    plt.imshow(viterbi_states[None, :], aspect="auto", cmap=cmap_trial,
               vmin=0, vmax=len(color_function(trial_lst_states)) - 1,
               extent=(0, time_bins, -lim, obs.shape[1] * lim), alpha=0.2)

    plt.xlim(0, obs.shape[0])
    plt.xlabel("frames")
    plt.yticks(lim * np.arange(obs.shape[1]), ['x{}'.format(d + 1) for d in range(obs.shape[1])])

    if kind == 0:
        plt.ylabel('observations before preprocessing')
        plt.title('observations {}, {}, {}, {}, {}s, {}'.format(name, trial_type, time_bins, trial_num_states,
                                                               delay, title))
        if save_figures:
            # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
            # exist
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/observations {}_{}_{}_{}_run{}.png'.format(name, trial_type, title,
                                                                           get_date(), run, bbox_inches="tight"))

    elif kind == 1:
        plt.ylabel('factors')
        plt.title('factors {}, {}, {}, {}, {}s, {}'.format(name, trial_type, time_bins, trial_num_states,
                                                           delay, title))
        if save_figures:
            # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
            # exist
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/factors {}_{}_{}_{}_run{}.png'.format(name, trial_type, title,
                                                                      get_date(), run, bbox_inches="tight"))
    elif kind == 2:
        plt.ylabel('components')
        plt.title('components {}, {}, {}, {}, {}s, {}'.format(name, trial_type, time_bins, trial_num_states,
                                                              delay, title))
        if save_figures:
            # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
            # exist
            new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
            subdirectory(file, new)
            plt.savefig(new + '/components {}_{}_{}_{}_run{}.png'.format(name, trial_type, title,
                                                                         get_date(), run, bbox_inches="tight"))

    return fig


def plot_transistion_matrix(hmm, time_bins, save_figures, name, trial_type, trial_num_states, file, type_fit, run):
    """
    function to plot the Transition Matrix and the  masked transition matrix (diagonal elements)
    :param hmm:         model
    :param time_bins:   number of frames
    :return:            fig
    """

    trans_matrix = hmm.transitions.transition_matrix

    fig = plt.figure(figsize=(6, 5))

    sns.heatmap(trans_matrix, cmap='viridis', alpha=0.8)
    plt.title('learned Transition Matrix: {}, {}, {}, {}, {}'.format(name, trial_type, time_bins, trial_num_states,
                                                           type_fit))
    plt.ylabel('state number')

    if save_figures:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/Trans.Mat_{}_{}_{}_run{}.png'.format(name, trial_type,
                                                                    get_date(), run, bbox_inches="tight"))


    return fig


def plot_transition_matrices(hmm, time_bins, save_figures, name, trial_type, trial_num_states, delay, file, type_fit,
                             kind_prepro, run):
    """
    function to plot the Transition Matrix and the  masked transition matrix (diagonal elements)
    :param hmm:                    fitted model
    :param time_bins:              number of frames for the viterbi algo
    :param save_figures:           true or false, depending if the figures have to be safed
    :param name:                   mouse name (eg. M5)
    :param trial_type:             trial typ (eg. Hit)
    :param trial_num_states:       number of hidden states (the model is fitted to)
    :param file:                   the current working directory
    :param run:                    integer: to prevent overwriting of runs with identical params
    :return:                       fig
    """

    trans_matrix = hmm.transitions.transition_matrix

    trans_matrix_zero_diag = np.copy(trans_matrix)
    np.fill_diagonal(trans_matrix_zero_diag, 0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0] = sns.heatmap(trans_matrix, ax=ax[0], cmap='viridis', alpha=0.8)
    ax[0].set_title('learned Transition Matrix:\n {}, {}, {}, {}, {}s, {}'.format(name, trial_type, time_bins,
                                                                                  trial_num_states, delay, type_fit))
    ax[0].set_ylabel('state number')

    ax[1] = sns.heatmap(trans_matrix_zero_diag,mask=trans_matrix_zero_diag == 0, ax=ax[1], cmap='viridis', alpha=0.8)
    ax[1].set_title('masked diagonal:\n same details as unmasked')
    ax[1].set_ylabel('state number')

    if save_figures:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/Trans.Mats_{}_{}_{}_{}_run{}.png'.format(name, trial_type, get_date(), kind_prepro, run,
                                                                     bbox_inches="tight"))

    return fig


def plot_transition_Matrices_1(hmm):

    trans_matrix = hmm.transitions.transition_matrix

    trans_matrix_zero_diag = np.copy(trans_matrix)
    np.fill_diagonal(trans_matrix_zero_diag, 0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0] = sns.heatmap(trans_matrix, ax=ax[0],
                        linewidth=0.1, cmap='viridis')

    ax[0].set_title('True Transition Matrix', y=1.2, transform=ax[0].transAxes)
    ax[0].xaxis.set_ticks_position('top')

    ax[1] = sns.heatmap(trans_matrix_zero_diag, ax=ax[1],
                        linewidth=0.1, cmap='viridis')

    ax[1].set_title('learned Transition Matrix', y=1.2, transform=ax[1].transAxes)
    ax[1].xaxis.set_ticks_position('top')

    return fig


def plot_posterior_prob(hmm, obs_vit, time_bins, trial_num_states, name, trial_type, delay, title, kind_prepro,
                        type_fit, save_figures, file, run):
    """
    function that calculates the posterior probability (alpha matrix), informing about the probability of a specific
    state at a given time step having seen the observations up to that that time step. This probabilities for each
    hidden state is plotted.
    :param hmm:                    fitted model
    :param obs_vit                 the observation data used for the viterbi algorithm
    :param time_bins:              number of frames for the viterbi algo
    :param save_figures:           true or false, depending if the figures have to be safed
    :param name:                   mouse name (eg. M5)
    :param trial_type:             trial typ (eg. Hit)
    :param trial_num_states:       number of hidden states (the model is fitted to)
    :param file:                   the current working directory
    :param title:                  typ of data (averaged or single, for viterbi)
    :param run:                    integer: to prevent overwriting of runs with identical params
    :return:                       fig
    """

    # posterior state probability, alpha matrix in forward pass
    posterior_probs = hmm.expected_states(obs_vit)[0]
    print('shape of the array holding the posterior probabilities of each hidden state\nand time step: {}'.format(np.shape(posterior_probs)))

    trial_lst_states = [i + 1 for i in range(0, trial_num_states)]
    # color function zur defintion der colors für  trial und true number of states
    colors_trial = color_function(trial_lst_states)

    xticks_lst = np.arange(0, time_bins + 1, step=10)

    fig = plt.figure(figsize=(12, 3.5), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use('seaborn')
    for k in range(trial_num_states):
        plt.plot(posterior_probs[:, k], label="State " + str(k + 1), lw=2,
                 color=colors_trial[k])

    plt.ylim((-0.01, 1.01))
    plt.xticks(xticks_lst)
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.xlabel("frames")
    plt.ylabel("p(state)")

    plt.title('posterior Probability: {}, {}, {}, {}, {}s ,{}, {}, {}'.format(name, trial_type, time_bins,
                                                                              trial_num_states, delay, title,
                                                                              kind_prepro, type_fit))

    if save_figures:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots results {} {} {}'.format(trial_type, time_bins, trial_num_states)
        subdirectory(file, new)
        plt.savefig(new + '/post_prob_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, kind_prepro,
                                                                        get_date(), run, bbox_inches="tight"))

    return posterior_probs, fig


