import numpy as np
from .settings import SUBJECTS_DIR
from os.path import join

r_mapping = {'O2': 0, 'O1': 1, 'P4': 2, 'P3': 3, 'C4': 4, 'C3': 5, 'F4': 6,
             'F3': 7, 'Fp2': 8, 'Fp1': 9, 'T6': 10, 'T5': 11, 'T4': 12,
             'T3': 13, 'F8': 14, 'F7': 15, 'Oz': 16, 'Pz': 17, 'Cz': 18,
             'Fz': 19, 'Fpz': 20, 'FT7': 21, 'FC3': 22, 'Fcz': 23, 'FC4': 24,
             'FT8': 25, 'TP7': 26, 'CP3': 27, 'Cpz': 28, 'CP4': 29, 'TP8': 30}


def extract(data_stream):
    raw_data = data_stream.readlines()
    data = []
    for line in raw_data:
        temp = np.array([float(x) for x in line.split()])
        data.append(temp)
    return np.array(data)


def format_fname(_group, _hand, _subj, _tr_num):
    directory = SUBJECTS_DIR
    fname_pattern = f'{_group}_subject_{_subj}_{_hand}_tr_{_tr_num}.dat'
    return join(directory, fname_pattern)


def extract_all(group=None, hand=None, subject=None, trial=None):
    hands = ('lefthand', 'righthand')
    groups = ('OLD', 'YOUNG')
    subjects = np.arange(1, 11)
    trials = np.arange(1, 16)
    epochs = []
    group = groups if group not in groups else (group, )
    hand = hands if hand not in hands else (hand, )
    subject = subjects if subject not in subjects else (subject, )
    trial = trials if trial not in trials else (trial, )
    for g in group:
        for h in hand:
            for s in subject:
                for t in trial:
                    fin = open(format_fname(g, h, s, t))
                    epochs.append(extract(fin))
                    fin.close()
    return np.array(epochs)


def select_channel(arr, ch):
    if ch in r_mapping.keys():
        ch = r_mapping[ch]
    elif ch not in r_mapping.values():
        raise ValueError('Wrong channel')
    mesh = [np.arange(arr.shape[i]) for i in range(len(arr.shape) - 2)] + [np.array([ch]), np.arange(arr.shape[-1])]
    ixgrid = np.ix_(*mesh)
    return np.squeeze(arr[ixgrid])