import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ssm
from preprocessing import get_date,  subdirectory


def AIC(num_states, obs_dim, likelihood):
    """
    This function calculates the  Akaike criterium (AIC), based on the number of FREE params (each row sums to 1!!) to
    take into account the Occram's Razor principle for model selection (how many hidden states):

#   AIC = 2ln(likelihood) - 2* num free params (--> min)

#   likelihood as hmm.log_probability(obs) is already a log.

    :param num_states:  number of hidden states on which the model is built
    :param obs_dim:     the number of ROI's (not dim reduced data), factors (FA dim reduced) or Components (PCA dim
                        reduced)
    :param likelihood:  the log likelihood calculated by the model
    :return:            a float number for that model/data
    """
    number_params = (np.square(num_states) - num_states) + (num_states * obs_dim) + (num_states - 1)
    AIC = 2 * number_params - 2 * likelihood

    return AIC


def get_input_parameters():

    max_num_states = int(input("Maximum number of hidden states:         "))
    num_loops_statistics = int(input("Number of loops:                         "))
    n_iters = int(input("Number of iterations in the fit:         "))

    emission_models = ['gaussian', 'autoregressive']
    while True:
        e_model = input("Emission model:                          ")
        if e_model not in emission_models:
            print('invalid input: gaussian or autoregressive' )
            continue
        else:
            break

    return max_num_states, num_loops_statistics, n_iters, e_model


def number_states(max_num_states):
    lst_number_states = [i + 1 for i in range(1, max_num_states)]
    return lst_number_states


def get_even_odds(obs):
    """
    function to separate the even form the odd time steps. Because the resulting number of observations are to low for
    the fitting (produces NAN's) the elements of the resulting arrays are repeated to get the original number of
    time steps again. The dimension, number of FA, components is also defined (needed for the fitting)
    :param obs:
    :return:
    """

    obs_even = obs[0:len(obs):2]
    obs_odd = obs[1:len(obs):2]

    # double the time steps because the serie is to short (NAN's)
    obs_even = np.repeat(obs_even, 2, axis=0)
    obs_odd = np.repeat(obs_odd, 2, axis=0)

    obs_dim = obs.shape[1]

    return obs_even, obs_odd, obs_dim


def nice_table(lst_dicts):

    lst_table = []
    for i in range(len(lst_dicts)):
        table = pd.DataFrame(lst_dicts[i], index=['e/o cross sum_llh normalized', 'llh normalized',
                                                  'sum_AIC normalized'])
        table = table.add_prefix('# st. =')
        lst_table.append(table)

    return lst_table


def loop_over_states_and_statistics(max_num_states, num_loops_statistics, N_iters, e_model, obs, verbose):
    """
    Inner loop: fit the model for the even and calc the log likelihood (llh) with that model for the odd and vice versa
    the metric for finding a good guess for the number of hidden states  is then the sum of these two
    log likelihoods (Lit: Celeux, Durand). The AIC criterion taking into account
    the Occram's razor principle (take the simples model which can be described with few params) based on the even-odd
    cross sum llh is also calculated and stored as well as the "normal" llh. The values are normalized. The codes
    for fitting and llh calculation are in the ssm package of the linderman's lab. The results is stored in a list which
    are the value of a dictionary ("dict_indicators"), the keys are the number of hidden states

    outer loop: loop for  statistics, do the inner loop n times to get n plots to compare (do the fit from other
    init values). Each inner loop produces a dictionary (keys: number of hidden states, values: list of llh's) These
    dictionaries are stored in a list (appended in each loop)

    :return: a list of dictionaries ("lst_dicts") containing the values of the e/o llh, AIC and llh for each number of
             hidden state   """

    lst_dicts = []
    lst_number_states = number_states(max_num_states)
    obs_even, obs_odd, obs_dim = get_even_odds(obs)

    # insert loop for statistics (gives in the end num_loop of  plots)
    for j in range(num_loops_statistics):

        dict_indicators = {}

        # loop over states
        for i in range(len(lst_number_states)):

            # fit model for the even and calc the log likeliood with that model(parmas) for the odd and vice versa
            # the indicator is then the sum of these two log likelihoods (Lit: Celeux, Durand)

            # open value list for dictionary of indicators
            lst = []

            even_hmm = ssm.HMM(lst_number_states[i], obs_dim,
                               observations=e_model, transitions="standard")

            # Fit
            hmm_lps = even_hmm.fit(obs_even, method="em", num_iters=N_iters, verbose=0)
            # llh_even = even_hmm.log_likelihood(obs_odd)

            odd_hmm = ssm.HMM(lst_number_states[i], obs_dim,
                              observations=e_model, transitions="standard")

            # Fit
            hmm_lps = odd_hmm.fit(obs_odd, method="em", num_iters=N_iters, verbose=0)
            # llh_odd = odd_hmm.log_likelihood(obs_even)

            # store sum of both (the criteria, normalized)
            sum_llh = even_hmm.log_likelihood(obs_odd) + odd_hmm.log_likelihood(obs_even)
            lst.append(sum_llh / np.size(obs, 0))

            # loglikelihood ohne cross validation
            hmm = ssm.HMM(lst_number_states[i], obs_dim,
                          observations="Gaussian", transitions="standard")

            # Fit (append the normalised value)
            hmm_lps = hmm.fit(obs, method="em", num_iters=N_iters, verbose=0)
            llh = hmm.log_likelihood(obs)
            lst.append(llh / np.size(obs, 0))

            # calculate AIC from cross sum llh (normalized value)
            lst.append(AIC(lst_number_states[i], obs_dim, sum_llh) / np.size(obs, 0))

            # insert the list as value into dict
            dict_indicators[lst_number_states[i]] = lst

        lst_dicts.append(dict_indicators)
        lst_table = nice_table(lst_dicts)

        print()
        print('loop {}:'.format(j))
        if verbose:
            print(lst_table[j].T)

    return lst_dicts, lst_table


def plots_of_eollh_llh_AIC_vs_statenumbers(lst_table, max_num_states, e_model, kind, obs, save_figure, name,
                                           trial_type, file, run):

    if e_model == 'autoregressive':
        e_model = 'autoreg.'

    loop = 0
    for item in lst_table:

        lst_y = [np.asarray(item.iloc[0].values), np.asarray(item.iloc[1].values)]
        label_y = ['e/o cross sum', 'normal']

        lst_y2 = [np.asarray(item.iloc[2].values)]
        label_y2 = 'AIC'

        txt = 'E-model: {}\nprepro.: {}'.format(e_model, kind)
        txt1 = '{}\n{} trials\ntime steps: {}\ndim: {}  '.format(name, trial_type, obs.shape[0], obs.shape[1])

        x = number_states(max_num_states)
        colors = ['r', 'g']

        fig, ax1 = plt.subplots(1,  figsize=(8,4))

        for n, colors in enumerate(colors):
            ax1.plot(x, lst_y[n], color=colors, label = label_y[n])
            ax1.set_xlabel('number of states')
            ax1.set_ylabel('log likelihood')
            ax1.set_title('log likelihoods/AIC \nfor different state numbers, loop {}, run {}'.format(loop, run))
            ax1.legend(bbox_to_anchor=(1.09, 1), loc=2, borderaxespad=0.0)
            ax1.text(x=1.11, y=0.11, transform=ax1.transAxes, s=txt, fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.15),
                     horizontalalignment='left',
                     verticalalignment='top')

            ax1.text(x=1.11, y=0.40, transform=ax1.transAxes, s=txt1, fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.15),
                     horizontalalignment='left',
                     verticalalignment='top')

            ax1.set_facecolor('whitesmoke')
            ax1.grid(color='lavender', which='both', linestyle='-', linewidth=1.8)
            fig.subplots_adjust(right=0.75)

            ax2 = ax1.twinx()
            ax2.plot(x, lst_y2[0], color='darkgrey', linestyle='--', label=label_y2)
            ax2.set_ylabel(label_y2)
            ax2.legend(bbox_to_anchor=(1.09, 0.85), loc=2, borderaxespad=0.0)

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots hyperparameter'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, get_date(), loop, kind, run,
                                                                     bbox_inches="tight"))

        loop += 1

    return fig


def plots_of_eollh_llh_AIC_vs_statenumbers_1(start, end, lst_table, max_num_states, e_model, kind, obs, save_figure,
                                             name, trial_type, file, run):

    if e_model == 'autoregressive':
        e_model = 'autoreg.'

    i = start
    end = end

    loop = i
    for i in range(i, end + 1):

        lst_y = [np.asarray(lst_table[i].iloc[0].values), np.asarray(lst_table[i].iloc[1].values)]
        label_y = ['e/o cross sum', 'normal']

        lst_y2 = [np.asarray(lst_table[i].iloc[2].values)]
        label_y2 = 'AIC'

        txt = 'E-model: {}\nprepro.: {}'.format(e_model, kind)
        txt1 = '{}\n{} trials\ntime steps: {}\ndim: {}  '.format(name, trial_type, obs.shape[0], obs.shape[1])

        x = number_states(max_num_states)
        colors = ['r', 'g']

        fig, ax1 = plt.subplots(1,  figsize=(8,4))

        for n, colors in enumerate(colors):
            ax1.plot(x, lst_y[n], color=colors, label = label_y[n])
            ax1.set_xlabel('number of states')
            ax1.set_ylabel('log likelihood')
            ax1.set_title('log likelihoods/AIC \nfor different state numbers, loop {}, run {}'.format(loop, run))
            ax1.legend(bbox_to_anchor=(1.09, 1), loc=2, borderaxespad=0.0)
            ax1.text(x=1.11, y=0.11, transform=ax1.transAxes, s=txt, fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.15),
                     horizontalalignment='left',
                     verticalalignment='top')

            ax1.text(x=1.11, y=0.40, transform=ax1.transAxes, s=txt1, fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.15),
                     horizontalalignment='left',
                     verticalalignment='top')

            ax1.set_facecolor('whitesmoke')
            ax1.grid(color='lavender', which='both', linestyle='-', linewidth=1.8)
            fig.subplots_adjust(right=0.75)

            ax2 = ax1.twinx()
            ax2.plot(x, lst_y2[0], color='darkgrey', linestyle='--', label=label_y2)
            ax2.set_ylabel(label_y2)
            ax2.legend(bbox_to_anchor=(1.09, 0.85), loc=2, borderaxespad=0.0)

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots hyperparameter'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, get_date(), loop, kind, run,
                                                                     bbox_inches="tight"))

        loop += 1

    return fig


def plots_of_eollh_llh_AIC_vs_statenumbers_2(lst_loops, lst_table, max_num_states, e_model, kind, obs, save_figure,
                                             name, trial_type, file, run):

    if e_model == 'autoregressive':
        e_model = 'autoreg.'

    lst_table_short = [lst_table[i] for i in lst_loops]
    i = 0
    for item in lst_table_short:

        loop = lst_loops[i]
        lst_y = [np.asarray(item.iloc[0].values), np.asarray(item.iloc[1].values)]
        label_y = ['e/o cross sum', 'normal']

        lst_y2 = [np.asarray(item.iloc[2].values)]
        label_y2 = 'AIC'

        txt = 'E-model: {}\nprepro.: {}'.format(e_model, kind)
        txt1 = '{}\n{} trials\ntime steps: {}\ndim: {}  '.format(name, trial_type, obs.shape[0], obs.shape[1])

        x = number_states(max_num_states)
        colors = ['r', 'g']

        fig, ax1 = plt.subplots(1,  figsize=(8,4))

        for n, colors in enumerate(colors):
            ax1.plot(x, lst_y[n], color=colors, label = label_y[n])
            ax1.set_xlabel('number of states')
            ax1.set_ylabel('log likelihood')
            ax1.set_title('log likelihoods/AIC \nfor different state numbers, loop {}, run {}'.format(loop, run))
            ax1.legend(bbox_to_anchor=(1.09, 1), loc=2, borderaxespad=0.0)
            ax1.text(x=1.11, y=0.11, transform=ax1.transAxes, s=txt, fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.15),
                     horizontalalignment='left',
                     verticalalignment='top')

            ax1.text(x=1.11, y=0.40, transform=ax1.transAxes, s=txt1, fontsize=10,
                     bbox=dict(facecolor='w', alpha=0.15),
                     horizontalalignment='left',
                     verticalalignment='top')

            ax1.set_facecolor('whitesmoke')
            ax1.grid(color='lavender', which='both', linestyle='-', linewidth=1.8)
            fig.subplots_adjust(right=0.75)

            ax2 = ax1.twinx()
            ax2.plot(x, lst_y2[0], color='darkgrey', linestyle='--', label=label_y2)
            ax2.set_ylabel(label_y2)
            ax2.legend(bbox_to_anchor=(1.09, 0.85), loc=2, borderaxespad=0.0)

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots hyperparameter'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, get_date(), loop, kind, run,
                                                                     bbox_inches="tight"))

        i += 1

    return fig


def get_hight_of_bars(Hit, CR, FA, Early, Miss, All, min, max):

    lst_Hit_bar = []
    for i in range(min, max+1):
         lst_Hit_bar.append(Hit[Hit == i].size)

    lst_CR_bar = []
    for i in range(min, max+1):
         lst_CR_bar.append(CR[CR == i].size)

    lst_FA_bar = []
    for i in range(min, max+1):
         lst_FA_bar.append(FA[FA == i].size)

    lst_Early_bar = []
    for i in range(min, max+1):
         lst_Early_bar.append(Early[Early == i].size)

    lst_Miss_bar = []
    for i in range(min, max + 1):
        lst_Miss_bar.append(Miss[Miss == i].size)

    lst_All_bar = []
    for i in range(min, max + 1):
        lst_All_bar.append(All[All == i].size)

    return lst_Hit_bar, lst_CR_bar, lst_FA_bar, lst_Early_bar, lst_Miss_bar, lst_All_bar


def bar_plot(mini, maxi, Hit, CR, FA, Early, Miss, All, name, kind, e_model, obs, save_figure, file, run):

    maxi = maxi + 1   # python counting
    ind = np.arange(maxi - mini)  # the x locations for the groups
    width = 0.5  # the width of the bars: can also be len(x) sequence

    # Höhe der Bars
    lst_Hit_bar = []
    for i in range(mini, maxi):
        lst_Hit_bar.append(Hit[Hit == i].size)

    lst_CR_bar = []
    for i in range(mini, maxi):
        lst_CR_bar.append(CR[CR == i].size)

    lst_FA_bar = []
    for i in range(mini, maxi):
        lst_FA_bar.append(FA[FA == i].size)

    lst_Early_bar = []
    for i in range(mini, maxi):
        lst_Early_bar.append(Early[Early == i].size)

    lst_Miss_bar = []
    for i in range(mini, maxi):
        lst_Miss_bar.append(Miss[Miss == i].size)

    lst_All_bar = []
    for i in range(mini, maxi):
        lst_All_bar.append(All[All == i].size)

    def autolabel(rects, ind):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax[ind].text(rect.get_x() + rect.get_width() / 2., 1. * height,
                         '%d' % int(height),
                         ha='center', va='bottom')

        return autolabel

    lst_bar_hights = [lst_Hit_bar, lst_CR_bar, lst_FA_bar, lst_Early_bar, lst_Miss_bar, lst_All_bar]
    lst_trial_typs = ['Hit', 'CR', 'FA', 'Early', 'Miss', 'All']

    fig, ax = plt.subplots(1, 6, figsize=(16, 4), sharey=True)

    for i in range(6):
        rect = ax[i].bar(ind, lst_bar_hights[i], width, color='g', alpha=0.5)
        ax[i].axhline(0, color='grey', linewidth=0.8)
        ax[0].set_ylabel('occurrence')
        ax[i].set_title(
            'Occurrence:\nNumbers of states\n{}, {}, {}\ntime steps {}\n{}'.format(lst_trial_typs[i], name, kind,
                                                                                   obs.shape[0], e_model))
        ax[i].set_xticks(ind)
        ax[i].set_xticklabels(([str(x) for x in range(mini, maxi)]))

        autolabel(rect, ind=i)

    if save_figure:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots hyperparameter'
        subdirectory(file, new)
        plt.savefig(new + '/{}_{}_{}_{}_{}_run{}.png'.format(name, kind, obs.shape[0], e_model, get_date(), run,
                                                             bbox_inches="tight"))

    return fig


def bar_plot_single(mini, maxi, arr, trial_type, name, kind, e_model, obs, save_figure, file, run):

    maxi = maxi + 1   # python counting
    ind = np.arange(maxi - mini)  # the x locations for the groups
    width = 0.5  # the width of the bars: can also be len(x) sequence

    # Höhe der Bars
    lst_arr_bar = []
    for i in range(mini, maxi):
        lst_arr_bar.append(arr[arr == i].size)

    def autolabel(rects, ax):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1. * height,
                         '%d' % int(height),
                         ha='center', va='bottom')

        return autolabel

    lst_bar_hights = lst_arr_bar.copy()

    fig, ax = plt.subplots(1, figsize=(4, 4), sharey=True)

    rect = ax.bar(ind, lst_bar_hights, width, color='g', alpha=0.5)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('occurrence')
    ax.set_title(
        'Occurrence:\nNumbers of states\n{}, {}, {}\ntime steps {}\n{}'.format(trial_type, name, kind, obs.shape[0],
                                                                               e_model))
    ax.set_xticks(ind)
    ax.set_xticklabels(([str(x) for x in range(mini, maxi)]))

    autolabel(rect, ax)

    if save_figure:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots hyperparameter'
        subdirectory(file, new)
        plt.savefig(new + '/{}_{}_{}_{}_{}_{}_run{}.png'.format(name, trial_type, kind, obs.shape[0], get_date(),
                                                                e_model, run, bbox_inches="tight"))

    return fig


def get_eo_and_AIC_values(lst_dicts, num_loops_statistics):

    lst_eo_all = []
    lst_AIC_all = []

    for i in range(num_loops_statistics):
        b = [*lst_dicts[i].values()]

        lst_eo = []
        lst_AIC = []

        for item in b:
            lst_eo.append(item[0])
            lst_AIC.append(item[2])

        lst_eo_all.append(lst_eo)
        lst_AIC_all.append(lst_AIC)

    print(lst_eo_all)
    print(lst_AIC_all)

    return lst_eo_all, lst_eo_all

def find_number_of_states(lst_dicts, number_loop):

    lst_values = [*lst_dicts[number_loop].values()]

    # get the eo and AIC values and convert to array
    lst_e_o = []
    lst_AIC = []

    for item in lst_values:
        lst_e_o.append(item[0])
        lst_AIC.append(item[2])

    array_AIC = np.array(lst_AIC)
    array_e_o = np.array(lst_e_o)

    #take minimum AIC, maximum eo
    min_AIC_index = int(np.where(array_AIC == min(array_AIC))[0])
    max_e_o_index = int(np.where(array_e_o == max(array_e_o))[0])

    number_state_min_AIC = min_AIC_index + 2
    number_state_max_e_o = max_e_o_index + 2

    probable_number_states = [number_state_max_e_o, number_state_min_AIC]

    # comparison between e_o_index and min_AIC_index
    if max_e_o_index == min_AIC_index:
        number_states = 2 + max_e_o_index
        probable_number_states.append(number_states)
        print('probable number of hidden states is: {}'.format(number_states))

    elif max_e_o_index != min_AIC_index:
        range_min = (max_e_o_index) + 2
        range_max = (min_AIC_index) + 2

        if max_e_o_index < min_AIC_index:
            print('probable number of hidden states are in the range of: {} to {}'.format(range_min, range_max))
            lst = [range_min, range_max]
            probable_number_states.append(lst)
        else:
            print('probable number of hidden states are in the range of: {} to {}'.format(range_max, range_min))
            lst = [range_max, range_min]
            probable_number_states.append(lst)

    return probable_number_states


def generate_list_number_states(probable_number_states):


    lst_number_states = []

    if probable_number_states[0] == probable_number_states[1]:
        lst_number_states.append(probable_number_states[0])

    return lst_number_states


def find_number_of_states_loop(lst_dicts, num_loops_statistics, trial_type):

    print('for each run for statistics:')
    lst_probable_number_states = []
    lst_input_statistics = []
    for i in range(num_loops_statistics):
        lst_values = [*lst_dicts[i].values()]

        # get the eo and AIC values and convert to array
        lst_e_o = []
        lst_AIC = []

        for item in lst_values:
            lst_e_o.append(item[0])
            lst_AIC.append(item[2])

        array_AIC = np.array(lst_AIC)
        array_e_o = np.array(lst_e_o)

        #take minimum AIC, maximum eo
        min_AIC_index = int(np.where(array_AIC == min(array_AIC))[0])
        max_e_o_index = int(np.where(array_e_o == max(array_e_o))[0])

        number_state_min_AIC = min_AIC_index + 2
        number_state_max_e_o = max_e_o_index + 2

        probable_number_states = [number_state_max_e_o, number_state_min_AIC]

        # comparison between e_o_index and min_AIC_index
        if max_e_o_index == min_AIC_index:
            number_states = 2 + max_e_o_index
            probable_number_states.append(number_states)
            print('probable number of hidden states is: {}'.format(number_states))

        elif max_e_o_index != min_AIC_index:
            range_min = (max_e_o_index) + 2
            range_max = (min_AIC_index) + 2

            if max_e_o_index < min_AIC_index:
                print('probable number of hidden states are in the range of: {} to {}'.format(range_min, range_max))
                lst = [range_min, range_max]
                probable_number_states.append(lst)
            else:
                print('probable number of hidden states are in the range of: {} to {}'.format(range_max, range_min))
                lst = [range_max, range_min]
                probable_number_states.append(lst)

        lst_probable_number_states.append(probable_number_states)
        lst_input_statistics.append(probable_number_states[0])

    print()
    print('the list taken as input for the bar plot:')
    print(lst_input_statistics)

    if trial_type == 'Hit':
        Hit = np.array(lst_input_statistics)
        return lst_probable_number_states, lst_input_statistics, Hit

    if trial_type == 'CR':
        CR = np.array(lst_input_statistics)
        return lst_probable_number_states, lst_input_statistics, CR

    if trial_type == 'FA':
        FA = np.array(lst_input_statistics)
        return lst_probable_number_states, lst_input_statistics, FA

    if trial_type == 'Early':
        Early = np.array(lst_input_statistics)
        return lst_probable_number_states, lst_input_statistics, Early

    if trial_type == 'Miss':
        Miss = np.array(lst_input_statistics)
        return lst_probable_number_states, lst_input_statistics, Miss


def find_number_of_states_loop1(lst_dicts, num_loops_statistics, trial_type):

    print('for each run for statistics:')

    lst_input_statistics = []
    for i in range(num_loops_statistics):
        lst_values = [*lst_dicts[i].values()]

        # get the eo and AIC values and convert to array
        lst_e_o = []
        lst_AIC = []

        for item in lst_values:
            lst_e_o.append(item[0])
            lst_AIC.append(item[2])

        array_AIC = np.array(lst_AIC)
        array_e_o = np.array(lst_e_o)

        #take minimum AIC, maximum eo
        min_AIC_index = int(np.where(array_AIC == min(array_AIC))[0])
        max_e_o_index = int(np.where(array_e_o == max(array_e_o))[0])

        number_state_min_AIC = min_AIC_index + 2
        number_state_max_e_o = max_e_o_index + 2


        # comparison between e_o_index and min_AIC_index
        if max_e_o_index == min_AIC_index:
            number_states = 2 + max_e_o_index
            lst_input_statistics.append(number_states)
            print('probable number of hidden states is: {}'.format(number_states))

        elif max_e_o_index != min_AIC_index:
            range_min = (max_e_o_index) + 2
            range_max = (min_AIC_index) + 2

            if max_e_o_index < min_AIC_index:
                print('probable number of hidden states are in the range of: {} to {}'.format(range_min, range_max))

            else:
                print('probable number of hidden states are in the range of: {} to {}'.format(range_max, range_min))

    print()
    print('the list taken as input for the bar plot:')
    print(lst_input_statistics)
    mini = min(lst_input_statistics)
    maxi = max(lst_input_statistics)
    print()
    print('the minimum number of states is: {}'.format(mini))
    print('the maximum number of states is: {}'.format(maxi))

    if trial_type == 'Hit':
        Hit = np.array(lst_input_statistics)
        return lst_input_statistics, Hit, mini, maxi

    if trial_type == 'CR':
        CR = np.array(lst_input_statistics)
        return lst_input_statistics, CR,  mini, maxi

    if trial_type == 'FA':
        FA = np.array(lst_input_statistics)
        return lst_input_statistics, FA,  mini, maxi

    if trial_type == 'Early':
        Early = np.array(lst_input_statistics)
        return  lst_input_statistics, Early,  mini, maxi

    if trial_type == 'Miss':
        Miss = np.array(lst_input_statistics)
        return lst_input_statistics, Miss,  mini, maxi

    if trial_type == 'All':
        All = np.array(lst_input_statistics)
        return lst_input_statistics, All,  mini, maxi


def get_mini_maxi_manual_case(Hit, CR, FA, Early, Miss, All):

    lst = [Hit, CR, FA, Early, Miss, All]

    lst_mini = []
    lst_maxi = []
    for item in lst:
        if len(item) > 1:
            lst_mini.append(min(item))
            lst_maxi.append(max(item))

    mini_m = min(lst_mini)
    maxi_m = max(lst_maxi)

    print('the minimum number of states over all trial types is: {}'.format(mini_m))
    print('the maximum number of states over all trial types is: {}'.format(maxi_m))

    return mini_m, maxi_m

