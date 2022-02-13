import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from fit_and_viterbi import color_function


# function für widget
def compare_widget(ts_base, ts_comp, obs, trial_num_states, posterior_probs, time_bins, name, trial_type, title,
                   kind_prepro, type_fit):

    plt.rcParams["axes.grid"] = False
    plt.style.use('seaborn')

    trial_lst_states = [i + 1 for i in range(0, trial_num_states)]
    colors_trial = color_function(trial_lst_states)

    tot_error = 0
    for i in range(0, 26):
        tot_error += np.abs(obs[ts_base, i] - obs[ts_comp, i])

    text = 'distance: {}'.format(np.round(tot_error, decimals=3))

    x = np.linspace(0, 26, num=27)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.scatter(x, obs[ts_base, :], color='b', label='ts: {}'.format(ts_base))
    ax.scatter(x, obs[ts_comp, :], color='r', label='ts: {}'.format(ts_comp))
    ax.legend()
    ax.set_xlabel('ROI')
    ax.set_ylabel('activity df/F')
    ax.set_title('dF/F  {}, {}, {}, {}, {}'.format(name, trial_type, time_bins, trial_num_states, title))

    ax.text(x=0.05, y=0.9, transform=ax.transAxes, s=text, fontsize=12, bbox=dict(facecolor='whitesmoke'),
            horizontalalignment='left',
            verticalalignment='top')

    xticks_lst = np.arange(0, time_bins + 1, step=10)

    fig = plt.figure(figsize=(12, 2.5), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use('seaborn')
    for k in range(trial_num_states):
        plt.plot(posterior_probs[:, k], label="State " + str(k), lw=2,
                 color=colors_trial[k])

    plt.xticks(xticks_lst)
    plt.vlines(x=ts_comp, ymin=0.0, ymax=1.0, colors='k', linestyles='dashed', label='ts_compare: {}'.format(ts_comp))
    plt.vlines(x=ts_base, ymin=0.0, ymax=1.0, colors='k', linestyles='dotted', label='ts_base: {}'.format(ts_base))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize=10)
    plt.xlabel("frames ")
    plt.ylabel("p(state)")
    plt.title('posterior Probability: {}, {}, {}, {}, {}, {}, {}'.format(name, trial_type, time_bins, trial_num_states,
                                                                        title, kind_prepro, type_fit))

    plt.show()


