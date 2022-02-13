import numpy as np
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


def get_date():
    """
    get the todays date
    :return:  todays date (string)
    """
    from datetime import date
    today = str(date.today())

    return today


def get_save_figure():
    """
    get the input about the question: do you want to save the figures?
    :return:  boolsche variable (True, False)
    """
    answer = ['yes', 'no']
    while True:
        save_figure = input("do you want to save the figure?    ")
        if save_figure not in answer:
            print('invalid input, answer either "yes" or "no"')
            continue
        else:
            break

    if save_figure == 'yes':
        save_figure = True


    else:
        save_figure = False

    return save_figure


def subdirectory(current, new):
    """
    :param new:     name of the new subdirectory of the current one, a new one is only created, if it does not
                    already exist
    """

    script_dir = os.path.dirname('current')
    new_dir = os.path.join(script_dir, new + '/')

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)


def subdirectory_1(file, new):
    """
    :param new:     name of the new subdirectory of the current one, a new one is only created, if it does not
                    already exist
    """

    script_dir = os.path.abspath(file)
    new_dir = os.path.join(script_dir, new + '/')

    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)

#the following funcions usually produce a print (to check the ongoings). To steer when the function should actually
#produce a print or just produce the return (for following calculations) a bool variable is included called:
#"prnt". If prnt = True --> a print is produced, if prnt = False --> no print ist produced

#General tasks

def get_and_check_trial_number(data):
    """
    function to get and check the wanted trial number
    :param data:    3 dimensional np.array "data" holding the dF/F of the selected Mouse/Sessions/Trial Typ)
    :return:        integer (index of the trial in the data array, checked and stored as a variable)
    """

    print('single trial:')

    while True:
        trial_index= input("trial number:                              ")
        trial_index = int(trial_index)
        if trial_index >= data.shape[0] :
            print('invalid input, trial number should be in range (0 to {})'.format(data.shape[0]))
            continue
        else:
            break

    return trial_index


def mean_data(data,prnt):
    """
    :param data: takes the 3 dimensional data array (trials, time_steps, dimensions) holding the dF/F values
    :return:     the mean of dF/F over the trials (axis 0 of 3 dim data array), a 2 dim array called "obs"
    """
    obs_aver = np.mean(data, axis=0)

    if prnt:
        print('averaged:')
        print('shape of the "obs_aver" array:            ', np.shape(obs_aver))
        print('number of time steps:                     ', obs_aver.shape[0])
        print('number of dimensions:                     ', obs_aver.shape[1])
        print()

    return obs_aver


def single_trial_data(trial_index, data, prnt):
    """
    function that returns the 2 dimensional np.array holding the dF/F for the selected trial index
    :param trial_index:   integer (index of the selected trial)
    :param data:          3 dim np.array holding the dF/F of the selected Mouse/sessions/trialtype
    :return:              2 dim np.array "obs_single" for the selected trial index (single trial)
    """

    obs_single = data[trial_index, :, :]

    if prnt:
        print('shape of the "obs_single" array:          ', np.shape(obs_single))
        print('number of time steps:                     ', obs_single.shape[0])
        print('number of dimensions:                     ', obs_single.shape[1])

    return obs_single


def generate_data_list(data, prnt):
    """
    :param data: takes the 3 dimensional data array (trials, time_steps, dimensions) holding the dF/F values
    :return:     returns the list of 2 dimensional trials (time_steps, dimensions)
    """

    lst_obs = []
    for i in range(0, data.shape[0]):
        lst_obs.append(data[i, :, :])

    if prnt:
        print('concatenated:')
        print('length of the list:                       ', len(lst_obs))
        print('shape of one element of the list:         ', np.shape(lst_obs[0]))

    return lst_obs


def concatenate(lst_obs, prnt):
    # stack this list with the original dimensions together an do the factor analysing on this stacked vector

    obs_conc = np.concatenate(lst_obs, axis=0)

    if prnt:

        print('shape of the concatenated array:')
        print('before dimensionality reduction           ', np.shape(obs_conc))
        print()

    return obs_conc


#Preprocessing with FA

def find_number_of_Factors_1(eigenval_limit, dimensions, obs,  kind, prnt):

    """this function calculates the number of factors with an Eigenvalue which is greater then the 'eigenval_limit,
    without the param trial_index
        :param   eigenval_limit: number (float) , recommended = 1.0
                 dimensions:     dimensions before dimensionality reduction (obs.shape[1])
                 obs:            2 dim array holding the averaged data
                 kind:           0, if data is averaged, 1 if data is single trial, 2 if data is concatenated
        :return: the number of factors generating the the data with eigenvalues greater then eigenval limit
                  """

    fa = FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                        method='minres', n_factors=dimensions, rotation=None, rotation_kwargs={},
                        use_smc=True)

    fa.fit(obs)
    eigenvals, x = fa.get_eigenvalues()

    # take the eigenvals >= 1 --> number of them = number of relevant factors
    num_FA_dim = len(eigenvals[eigenvals >= eigenval_limit])

    if prnt:

        if kind == 0:
            print('averaged:')
            print('Number of Factors:                           ', num_FA_dim)


        elif kind == 2:
            print('concatenated:')
            print('Number of Factors:                           ', num_FA_dim)

    return num_FA_dim


def find_number_of_Factors(eigenval_limit, dimensions, obs, trial_index,  kind, prnt):

    """this function calculates the number of factors with an Eigenvalue which is greater then the 'eigenval_limit
        :param   eigenval_limit: number (float) , recommended = 1.0
                 dimensions:     dimensions before dimensionality reduction (obs.shape[1])
                 obs:            2 dim array holding the averaged data
                 kind:           0, if data is averaged, 1 if data is single trial, 2 if data is concatenated
        :return: the number of factors generating the the data with eigenvalues greater then eigenval limit
                  """

    fa = FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                        method='minres', n_factors=dimensions, rotation=None, rotation_kwargs={},
                        use_smc=True)

    fa.fit(obs)
    eigenvals, x = fa.get_eigenvalues()

    # take the eigenvals >= 1 --> number of them = number of relevant factors
    num_FA_dim = len(eigenvals[eigenvals >= eigenval_limit])

    if prnt:

        if kind == 0:
            print('averaged:')
            print('Number of Factors:                           ', num_FA_dim)

        elif kind == 1:
            print('single trial:')
            print('trial number:                                ', trial_index)
            print('Number of Factors:                           ', num_FA_dim)

        elif kind == 2:
            print('concatenated:')
            print('Number of Factors:                           ', num_FA_dim)

    return num_FA_dim




def factors(num_FA_dim, obs, kind, prnt):
    """
    Does the Factor anlaysing/dimensionality reduction
    :param num_FA_dim:  the number of factors generating the the data with eigenvalues greater then eigenval limit
                        (integer)
    :param obs:         data to be generated by the factors (2d np.array)
    :param: kind:       0,1 or 2 depending on the data: averaged data: kind = 0, single Trial: kind = 1
                        concatenated data: kind = 2
    :return:            the factors generating the data with less dimensions
                        (2d np.array, shape:(time steps, num_FA_dim)
    """

    fa = FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                        method='minres', n_factors=num_FA_dim, rotation=None, rotation_kwargs={},
                        use_smc=True)

    fa.fit(obs)
    obs_transformed_FA = fa.transform(obs)

    if prnt:

        if kind == 0:
            print('shape of the "obs_transformed_FA" array:     ', np.shape(obs_transformed_FA))
            print('number of time steps:                        ', obs_transformed_FA.shape[0])
            print('number of dimensions:                        ', obs_transformed_FA.shape[1])
            print()

        elif kind == 1:
            print('shape of the "obs_transformed_FA_s" array:   ', np.shape(obs_transformed_FA))
            print('number of time steps:                        ', obs_transformed_FA.shape[0])
            print('number of dimensions:                        ', obs_transformed_FA.shape[1])
            print()

        elif kind == 2:
            print('shape of the "obs_transformed_FA_c" array:   ', np.shape(obs_transformed_FA))
            print('number of time steps:                        ', obs_transformed_FA.shape[0])
            print('number of dimensions:                        ', obs_transformed_FA.shape[1])
            print()

        return obs_transformed_FA


def factors_lst(number_factors, lst_obs, prnt):
    """
    Does the Factor anlaysing/dimensionality reduction for a list of observations with a loop ober that list
    :param number_factors:  the number of factors to be taken (the reduced dimensionality), has to be the same for
                            all the list elements (integer)
    :param lst_obs:         list,  elements hold the dF/F for the Mouse/sessions/trial type of one specific trial
    :return:                list, elements hold the transformed observations. the shape of the elements is now
                            (time steps, number of factors)
    """

    lst_obs_transformed = []

    for item in lst_obs:

        fa = FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                            method='minres', n_factors=number_factors, rotation=None, rotation_kwargs={},
                            use_smc=True)

        fa.fit(item)
        obs_transformed = fa.transform(item)

        lst_obs_transformed.append(obs_transformed)

    if prnt == True:

        print()
        print('shape of one element of the list')
        print('after the dim reduction:                  ', np.shape(lst_obs_transformed[0]))
        print('number of time steps:                     ', lst_obs_transformed[0].shape[0])
        print('number of dimensions:                     ', lst_obs_transformed[0].shape[1])

    return lst_obs_transformed


#Preprocessing with PCA

def corr_matrix(obs,kind,prnt):
    """
    function to calculate the correlation matrix of the obs (obs_s) array.
    :param obs: 2dim array holding the dF/F averaged or single trial, has to be transposed to get (27,27)
    :return:    2dim square array (27, 27)
    """
    # corr matrix in dim direction
    corr_mat = np.corrcoef(obs.T)

    if prnt:
        if kind == 0:
            print('averaged:')
            print('shape of the correlation matrix of the "obs_aver" array:                  ',np.shape(corr_mat))

        elif kind == 2:
            print('concatenated:')
            print('shape of the correlation matrix of the "obs_conc" array:                  ', np.shape(corr_mat))

    return corr_mat

def corr_matrix_1(obs,trial_index, kind, prnt):
    """
    function to calculate the correlation matrix of the obs (obs_s) array. for single trial mit trial index
    :param obs: 2dim array holding the dF/F averaged or single trial, has to be transposed to get (27,27)
    :return:    2dim square array (27, 27)
    """
    # corr matrix in dim direction
    corr_mat = np.corrcoef(obs.T)

    if prnt:
        if kind == 0:
            print('averaged:')
            print('shape of the correlation matrix of the "obs_aver" array:                  ',np.shape(corr_mat))
        elif kind == 1:
            print('single trial {}:'. format(trial_index))
            print('shape of the correlation matrix of the "obs_single" array:                ', np.shape(corr_mat))
        elif kind == 2:
            print('concatenated:')
            print('shape of the correlation matrix of the "obs_conc" array:                  ', np.shape(corr_mat))

    return corr_mat


def eigen_sorted(mat, kind, prnt):
    eigen_vals, eigen_vecs = np.linalg.eig(mat)
    sorted_index = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[sorted_index]
    # similarly sort the eigenvectors
    eigen_vecs = eigen_vecs[:,sorted_index]

    if prnt:
        if kind == 0:
            print('sorted Eigenvalues for averaged "obs_aver":\n {}'.format(eigen_vals))
        elif kind == 1:
            print('sorted Eigenvalues for single trial "obs_single":\n {}'.format(eigen_vals))
        elif kind == 2:
            print('sorted Eigenvalues for single trial "obs_conc":\n {}'.format(eigen_vals))

    return eigen_vals, eigen_vecs


def explained(eigen_vals):
    # explained variance plot daten erstellen
    tot = sum(eigen_vals)
    var_exp = [i / tot for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    return cum_var_exp, var_exp

def number_of_main_components(tot_expl_variance, obs, kind, prnt):
    """

    :param tot_explained_variance: float (number between 0 and 1), desired explained variance, indirectly choosing
                                   the number of dimensions after the dimensionality reduction
                                   recommended: 0.99 --> leading to 4 components for Yasir's data

    :return:                       integer, number of components needed to explain the variance
    """
    mat = corr_matrix(obs,kind, prnt=False)
    eigen_vals = eigen_sorted(mat,kind, prnt=False)[0]
    cum_var_exp = explained(eigen_vals)[0]
    n_components = len(cum_var_exp[cum_var_exp <= tot_expl_variance])

    if prnt:
        if kind == 0:
            print('number of components to get the total explained variance, obs_aver:       ', n_components)
        elif kind == 1:
            print('number of components to get the total explained variance, "obs_single":   ', n_components)
        elif kind == 2:
            print('number of components to get the total explained variance, "obs_conc":     ',n_components)
    print()
    return n_components


def pca(n_components, obs, kind, prnt):
    """
    function returning the dimension reduced data array using PCA with python package sklearn
    :param n_components:  integer, how many components are needed for the requested explained variance
    :param obs:           2d array, data to be transformed (to be reduced in dimensions),
                          shape (times steps, dimensions) dimensions = 27 in Yasirs data

    :return:              2d array, transformed data,
                          shape (time steps, n_components) n_components < dimensions
    """

    # standardize features
    sc = StandardScaler()
    obs_std_skl = sc.fit_transform(obs)

    # transform
    pca = PCA(n_components)
    obs_transformed_PCA = pca.fit_transform(obs_std_skl)

    if prnt == True:
        print()

        if kind == 0:
            print('shape of the "obs_transformed_PCA" array:     ', np.shape(obs_transformed_PCA))
            print('number of time steps:                         ', obs_transformed_PCA.shape[0])
            print('number of dimensions:                         ', obs_transformed_PCA.shape[1])

        elif kind == 1:
            print('shape of the "obs_transformed_PCA_s" array:   ', np.shape(obs_transformed_PCA))
            print('number of time steps:                         ', obs_transformed_PCA.shape[0])
            print('number of dimensions:                         ', obs_transformed_PCA.shape[1])

        elif kind == 2:
            print('shape of the "obs_transformed_PCA_c" array:   ', np.shape(obs_transformed_PCA))
            print('number of time steps:                         ', obs_transformed_PCA.shape[0])
            print('number of dimensions:                         ', obs_transformed_PCA.shape[1])

        return obs_transformed_PCA

def plot_explained_variance(tot_expl_variance, obs, trial_type, name, save_figure, file, title, run, ):
    """
    function to plot the individual and the cumulative explained variance
    :param tot_expl_variance:     input of the requested explained variance
    :param: obs                   2d array holding the averaged dF/F
                                   for single trial --> obs_single
                                   for concatened --> obs_conc

    :param: obs_dim_before         dimension of the 2d array obs/obs_single over axis 1 (dimensions before reduction)
    :param: trial_typ              trial typ (input)
    :param: title                  string: title of the figure (obs, obs_single, obs_conc)
    :param: run                    integer to avoid overwrititing the file holding the plot
    :param: save_figures           if save_figures = true, figure will be saved in the same folder wiht the given name
    :return:                       plot of the individual and cumulative explained variance
    """

    # get the subset of eigenvectors belonging to the main compenents using the eigen_sorted function returning
    # the eigenvectors in the second place
    n_components = number_of_main_components(tot_expl_variance, obs, kind=0, prnt=False)
    mat = corr_matrix(obs, kind=0, prnt=False)
    eigen_vecs_subset = eigen_sorted(mat, kind=0, prnt=False)[1][:, 0:n_components]

    print('shape of the array containing the main eigenvectors:      ', np.shape(eigen_vecs_subset))

    # plot the variances

    y = np.ones(obs.shape[1]) * tot_expl_variance
    eigen_vals = eigen_sorted(mat, kind=0, prnt=False)[0]
    var_exp = explained(eigen_vals)[1]
    cum_var_exp = explained(eigen_vals)[0]

    fig = plt.figure()
    plt.bar(range(1, obs.shape[1] + 1), var_exp, alpha=0.5, align='center', label='individual expl. variance')
    plt.step(range(1, obs.shape[1] + 1), cum_var_exp, where='mid', label='cumulative expl. variance')
    plt.plot(range(1, obs.shape[1] + 1), y, linestyle='--', color='k', alpha=0.5)
    plt.vlines(n_components, 0, 1, linestyle='--', color='k', alpha=0.5)
    plt.xlabel('Number of Components')
    plt.ylabel('ind/cum expl. Variance')
    plt.legend()
    plt.title('{}, {}, {},\n individual expl. Variance/cumulative expl. Variance'.format(name, trial_type, title))

    if save_figure:
        # create new subdirectory name, function subdirectory creates that subdirectory if it does not already
        # exist
        new = name + ' plots preprocessed'
        subdirectory(file, new)
        plt.savefig(new + '/{}_{}_{}_PCA_{}_run{}.png'.format(name, trial_type, title, get_date(),
                                                              run, bbox_inches="tight"))

    return fig
