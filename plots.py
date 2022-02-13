import numpy as np
import matplotlib.pyplot as plt
from preprocessing import get_date,  subdirectory


def plot_data(obs, name, trial_type, frames, delay, trial_index, save_figure, file, title, run, kind_data, kind_prepro):
    """
    plots the data (2d array, dF/F) before proprocessing
    :param obs:           2d array
    :param name:          string, mouse (M5, M6 etc)
    :param trial_type     string, trial type (Hit, FA etc)
    :param frames         integer number of time steps
    :param title          string, which data (averaged, single trial, conc)
    :param run            integer, run number to avoid overwriting
    :param save_figure    boolsche variable (True if input = 'yes', False otherwise)
    :param trial_index    integer, index of single trial
    :param file           current working directory (function: get_file_and_label )
    :param title          string, which type of data: average, single or concatenated
    :param kind_data      integer, 0,1,2 depending on type of data (0=average, 1=single, 2=conc)
    :param kind_prepro    string, 'nil' for no preprocessing, 'FA' for FA, 'PCA' for PCA
    :return:              figure
    """

    lim = 1.05 * abs(obs).max()

    fig = plt.figure(figsize=(10, 10))
    for d in range(obs.shape[1]):
        plt.plot(obs[: ,d] + lim * d, '-k')

    plt.xlim(0, obs.shape[0])
    plt.xlabel("time steps")

    plt.yticks(lim * np.arange(obs.shape[1]), ['x{}'.format(d + 1) for d in range(obs.shape[1])])

    if kind_data == 0:
        if kind_prepro == 'nil':
            plt.ylabel('observations')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}, '.format(name, trial_type, frames, delay, title))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots data'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, get_date(),
                                                                  run, bbox_inches="tight"))
        elif kind_prepro == 'FA':
            plt.ylabel('factors')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}, FA'.format(name, trial_type, frames, delay,  title))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots preprocessed'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_FA_run{}.png'.format(name, trial_type, title, get_date(),
                                                                     run, bbox_inches="tight"))
        elif kind_prepro == 'PCA':
            plt.ylabel('components')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}, PCA'.format(name, trial_type, frames, delay, title))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots preprocessed'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_PCA_run{}.png'.format(name, trial_type, title, get_date(),
                                                                     run, bbox_inches="tight"))

    elif kind_data == 1:
        title_plot = 'single trial, number: {}'.format(trial_index)
        if kind_prepro == 'nil':
            plt.ylabel('observations')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}'. format(name, trial_type, frames, delay, title_plot))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots data'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, get_date(),
                                                                  run, bbox_inches="tight"))
        elif kind_prepro == 'FA':
            plt.ylabel('factors')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}, FA'.format(name, trial_type, frames, delay, title_plot))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots preprocessed'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_FA_run{}.png'.format(name, trial_type, title, get_date(),
                                                                     run, bbox_inches="tight"))
        elif kind_prepro == 'PCA':
            plt.ylabel('components')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}, PCA'.format(name, trial_type, frames, delay,
                                                                                title_plot))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots preprocessed'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_PCA_run{}.png'.format(name, trial_type, title, get_date(),
                                                                      run, bbox_inches="tight"))

    elif kind_data == 2:
        if kind_prepro == 'nil':
            plt.ylabel('observations')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}'. format(name, trial_type, frames, delay, title))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots data'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, get_date(),
                                                                  run, bbox_inches="tight"))
        elif kind_prepro == 'FA':
            plt.ylabel('factors')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}, FA'.format(name, trial_type, frames, delay, title))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots preprocessed'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_FA_run{}.png'.format(name, trial_type, title, get_date(),
                                                                  run, bbox_inches="tight"))
        elif kind_prepro == 'PCA':
            plt.ylabel('components')
            plt.title('{}, {} trials, {} time steps, {}s delay, {}, PCA'.format(name, trial_type, frames, delay, title))

            if save_figure:
                # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
                # exist
                new = name + ' plots preprocessed'
                subdirectory(file, new)
                plt.savefig(new + '/{}_{}_{}_{}_PCA_run{}.png'.format(name, trial_type, title, get_date(),
                                                                      run, bbox_inches="tight"))
    return fig


def plot_data_FA(obs, name, trial_type, frames, delay, save_figure, file, title, run):
    """
    function to plot the preprocessed data (averaged and concatenated), for individual selection of average,
    concatenated. This is necessary because the plot_data function has as argument the trial index which could be
     missing if only averaged
    :param obs:
    :param name:
    :param trial_type:
    :param frames:
    :param save_figure:
    :param file:
    :param title:
    :param run:
    :return:
    """

    lim = 1.05 * abs(obs).max()

    fig = plt.figure(figsize=(10, 10))
    for d in range(obs.shape[1]):
        plt.plot(obs[:, d] + lim * d, '-k')

    plt.xlim(0, obs.shape[0])
    plt.xlabel("time steps")

    plt.yticks(lim * np.arange(obs.shape[1]), ['x{}'.format(d + 1) for d in range(obs.shape[1])])

    plt.ylabel('factors')
    plt.title('{}, {} trials, {} time steps, {}s delay, {}, FA'.format(name, trial_type, frames, delay, title))

    if save_figure:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots data'
        subdirectory(file, new)
        plt.savefig(new + '/{}_{}_{}_{}_FA_run{}.png'.format(name, trial_type, title, get_date(),
                                                          run, bbox_inches="tight"))

    return fig


def plot_data_PCA(obs, name, trial_type, frames, delay, save_figure, file, title, run):
    """
    function to plot the preprocessed data (averaged and concatenated), for individual selection of average,
    concatenated. This is necessary because the plot_data function has as argument the trial index which could be
     missing if only averaged
    :param obs:
    :param name:
    :param trial_type:
    :param frames:
    :param save_figure:
    :param file:
    :param title:
    :param run:
    :return:
    """

    lim = 1.05 * abs(obs).max()

    fig = plt.figure(figsize=(10, 10))
    for d in range(obs.shape[1]):
        plt.plot(obs[:, d] + lim * d, '-k')

    plt.xlim(0, obs.shape[0])
    plt.xlabel("time steps")

    plt.yticks(lim * np.arange(obs.shape[1]), ['x{}'.format(d + 1) for d in range(obs.shape[1])])

    plt.ylabel('components')
    plt.title('{}, {} trials, {} time steps, {}s delay, {}, PCA'.format(name, trial_type, frames, delay, title))

    if save_figure:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots data'
        subdirectory(file, new)
        plt.savefig(new + '/{}_{}_{}_{}_PCA_run{}.png'.format(name, trial_type, title, get_date(),
                                                          run, bbox_inches="tight"))

    return fig


def plot_data_1(obs, name, trial_type, frames, delay, save_figure, file, title, run):
    """
    function to plot the prepared data for individual selection for averaged and concatenated data. This is
    necessary because the plot_data function has as argument the trial index which could be missing if only averaged
    or concatenated data is used. For single trial this is optional, also plot_data can be used
    :param obs:
    :param name:
    :param trial_type:
    :param frames:
    :param save_figure:
    :param file:
    :param title:
    :param run:
    :return:
    """

    lim = 1.05 * abs(obs).max()

    fig = plt.figure(figsize=(10, 10))
    for d in range(obs.shape[1]):
        plt.plot(obs[:, d] + lim * d, '-k')

    plt.xlim(0, obs.shape[0])
    plt.xlabel("time steps")

    plt.yticks(lim * np.arange(obs.shape[1]), ['x{}'.format(d + 1) for d in range(obs.shape[1])])

    plt.ylabel('observations')
    plt.title('{}, {} trials, {} time steps, {}s delay, {}'.format(name, trial_type, frames, delay, title))

    if save_figure:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots data'
        subdirectory(file, new)
        plt.savefig(new + '/{}_{}_{}_{}_run{}.png'.format(name, trial_type, title, get_date(),
                                                          run, bbox_inches="tight"))

    return fig

