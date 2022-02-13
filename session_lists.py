import numpy as np


#mouse 5:
lst_sessions_180_M5 = ['2017_03_13_session01','2017_03_13_session02','2017_03_14_session02',
                       '2017_03_15_session01','2017_03_16_session01','2017_03_16_session02',
                       '2017_03_16_session03','2017_03_17_session01','2017_03_22_session01',
                       '2017_03_22_session03','2017_03_22_session04','2017_03_23_session01',
                       '2017_03_23_session02','2017_03_24_session01','2017_03_24_session02',
                       '2017_03_24_session03','2017_03_28_session01','2017_03_29_session01','2017_03_29_session02']

lst_sessions_220_M5 = ['2017_03_06_session01','2017_03_06_session02' ]


#mouse 6 (to get the lists: head in notebooks of M6)
lst_sessions_180_M6 = ['20170925_a', '20170925_b', '20170925_c']

lst_sessions_200_M6 = ['20170926_a']

lst_sessions_220_M6 = ['20170927_a', '20170927_b', '20170927_c', '20171004_a', '20171005_a',
                       '20171005_b', '20171006_b', '20171009_a', '20171009_b', '20171010_a',
                       '20171010_b', '20171010_c', '20171011_a', '20171012_b', '20171012_c']

lst_sessions_240_M6_5 = ['20170926_b', '20170926_c','20170928_a' , '20170929_a']

lst_sessions_240_M6_6 = ['20170928_b', '20170929_b','20171002_a', '20171002_c','20171011_b']

lst_sessions_240_M6_7 = ['20170929_c', '20171011_c']


#mouse 7 (to get the lists: head in notebooks of M7)
lst_sessions_160_M7 = ['20171113_a', '20171114_a', '20171114_b', '20171114_c', '20171115_a', '20171115_b',
                       '20171115_c', '20171115_d', '20171122_a', '20171122_b', '20171122_c', '20171122_d',
                       '20171123_a', '20171123_b', '20171127_a', '20171127_b', '20171127_c', '20171128_a',
                       '20171128_b', '20171129_a', '20171129_b', '20171129_d', '20171129_e', '20171130_a',
                       '20171130_b', '20171130_c']

lst_sessions_180_M7 = ['20171116_b',]


#mouse 9 (to get the lists: head in notebooks of M9)
lst_sessions_180_M9 = ['20180301_a', '20180301_b', '20180302_a', '20180302_b', '20180302_c', '20180305_a',
                       '20180305_b', '20180306_a', '20180306_b', '20180307_a', '20180307_b', '20180307_c',
                       '20180308_a', '20180308_b', '20180309_a', '20180309_b', '20180313_a', '20180313_b',
                       '20180313_c', '20180315_a', '20180316_a', '20180316_b']

lst_sessions_180_M9_red = ['20180301_a', '20180301_b', '20180302_a', '20180302_b', '20180302_c',
                           '20180313_b', '20180313_c', '20180315_a', '20180316_a', '20180316_b']


def get_lst_sessions(name, frames, delay, trial_type):

    """
    In this function the list of the sessions which are examined is generated

    :param name:     input 'M5', 'M6', 'M7', 'M9'
    :param frames:   input 160, 180, 200, 220, 240 (according to the possible lenght of the sessions)
    :param delay:    input 5,6,7  (for M6 240 (sessions with different delay times)
    :return:         a list of sessions from which the data is taken
    """
    if name == 'M5' and frames == 220:
        lst_sessions = lst_sessions_220_M5

    elif name == 'M5' and frames == 180:
        lst_sessions = lst_sessions_180_M5

    elif name == 'M6' and frames == 180:
        lst_sessions = lst_sessions_180_M6

    elif name == 'M6' and frames == 200:
        lst_sessions = lst_sessions_200_M6

    elif name == 'M6' and frames == 220:
        lst_sessions = lst_sessions_220_M6

    elif name == 'M6' and frames == 240 and delay == 5:
        lst_sessions = lst_sessions_240_M6_5

    elif name == 'M6' and frames == 240 and delay == 6:
        lst_sessions = lst_sessions_240_M6_6

    elif name == 'M6' and frames == 240 and delay == 7:
        lst_sessions = lst_sessions_240_M6_7

    elif name == 'M7' and frames == 160:
        lst_sessions = lst_sessions_160_M7

    elif name == 'M7' and frames == 180:
        lst_sessions = lst_sessions_180_M7

    elif name == 'M9' and frames == 180:
        if trial_type == 'Hit':
            lst_sessions = lst_sessions_180_M9_red
        else:
            lst_sessions = lst_sessions_180_M9

    else:
        return 'invalid input, check Mouse name and possible number of frames'

    #print('the list to be analysed is:')
    #print()
    #print(lst_sessions)
    #print()
    #print('the length of the list is:', len(lst_sessions))

    print_lst_sessions(lst_sessions)

    return lst_sessions


def print_lst_sessions(lst_sessions):
    print('the list to be analysed is:')
    print()
    print(lst_sessions)
    print()
    print('the length of the list is:', len(lst_sessions))


def get_list_sessions_2(*args):

    """
    function to create your own list of sessions (combinations) and start a list to collect lists for other mice
    :param  lst_sessions_all: is a list, opened at the very beginning of the main code to collect the lst_sessions
                              for different mice
    :param args: name of lists to combine for one mouse
    :return: lst_sessions
    """
    lst_sessions = []
    for item in args:
        lst_sessions = lst_sessions + item

    print_lst_sessions(lst_sessions)


    return lst_sessions


def flatten_list_sessions_all(lst):
    """"""
    lst_f = [item for sublist in lst for item in sublist]
    return lst_f



def get_file_and_label(name):
    """
    :param name:  'M5', 'M6', 'M7', 'M9',  taken from input
    :return:      file to generate the path to the data, string to define the path to the H5py.File
                  mouse_x_ (string), to build the correct csv file name in which the metadata (trial typ) is stored
    """
    if name == 'M5':
        file = 'Yasir data mouse 5/mou_5.h5'
        label = 'mouse_5_'

    elif name == 'M6':
        file = 'Yasir data mouse 6/mou_6.h5'
        label = 'mouse_6_'

    elif name == 'M7':
        file = 'Yasir data mouse 7/mou_7.h5'
        label = 'mouse_7_'

    elif name == 'M9':
        file = 'Yasir data mouse 9/mou_9.h5'
        label = 'mouse_9_'

    return file, label


def get_path(path, file):
    path = path + file
    print('complete path to the original H5py datas:')
    print()
    print(path)
    print()
    print()

    return path




def get_and_check_trial_type():

        trial_types = ['Hit', 'CR', 'FA', 'Early', 'Miss', 'All']
        while True:
            trial_type = input("Trial type:         ")
            if trial_type not in trial_types:
                print('invalid input, check trial type')
                continue
            else:
                break

        return trial_type


def get_and_check_mouse_name():

    names = ['M5', 'M6', 'M7', 'M9']
    while True:
        name = input("Mouse name:         ")
        if name not in names:
            print('invalid input, check mouse name')
            continue
        else:
            break
    return name

def get_and_check_frame_number(name):
    if name == 'M5':
        numbers = [180, 220]
        while True:
            frames = input("Number of frames:   ")
            frames = int(frames)
            if frames not in numbers:
                print('invalid input, frames/time steps for M5')
                continue
            else:
                break

        return frames

    if name == 'M6':
        numbers = [180, 200, 220, 240 ]
        while True:
            frames = input("Number of frames:   ")
            frames = int(frames)
            if frames not in numbers:
                print('invalid input, frames/time steps for M6')
                continue
            else:
                break
        return frames

    if name == 'M7':
        numbers = [160, 180]
        while True:
            frames = input("Number of frames:   ")
            frames = int(frames)
            if frames not in numbers:
                print('invalid input, frames/time steps for M7')
                continue
            else:
                break

        return frames

    if name == 'M9':
        numbers = [180]
        while True:
            frames = input("Number of frames:   ")
            frames = int(frames)
            if frames not in numbers:
                print('invalid input, frames/time steps for M9')
                continue
            else:
                break
        return frames


def get_and_check_delay(name, frames):
    if name == 'M6' and frames == 240:
        delays = [5,6,7]
        while True:
            delay = input("delay in seconds:   ")
            delay = int(delay)
            if delay not in delays:
                print('invalid input, check possible delays in seconds for M6 with 240 frames')
                continue
            else:
                break
        return delay
    else:
        delay = 0
        return delay


def get_and_check_delay_all(name, frames):
    delays = [1.1,2,3,4,5,6,7]
    while True:
        delay = input("delay in seconds:   ")
        delay = float(delay)
        if delay not in delays:
            print('invalid input, check possible delays in seconds')
            continue
        else:
            break

    return delay


def lst_sessions_M5(frames):

    if frames == 180:
        return lst_sessions_180_M5

    if frames == 220:
        return lst_sessions_220_M5

    if frames != 180 and frames != 220:
        return 'no such sessions available'