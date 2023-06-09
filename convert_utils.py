import random, os, sys
import numpy as np
import utils
from my_arrays import find_runs
from my_os import recursive_file_retrieval


def get_gender_lists(SVC_data_dir):
    print('Getting gender info...')
    performer_gender_list = utils.get_damp_gender(ignore_unknowns=True)
    _, spmel_fps = recursive_file_retrieval(SVC_data_dir)
    spmel_perf_id = [os.path.basename(fp).split('_')[0] for fp in spmel_fps]
    performance_gender_subset = [perf_gen for perf_gen in performer_gender_list if perf_gen[0] in spmel_perf_id]
    females = [perf for perf, gend in performance_gender_subset if gend == ' F']
    males = [perf for perf, gend in performance_gender_subset if gend == ' M']
    gender_separated_lists = [males, females]
    return gender_separated_lists


class NoMatchError(Exception):
    pass


def get_relevant_avg_pitches(continuous_pitch_feats, window_size, irrelenvant_ind=0):
    average_pitches = []
    for idx in range(len(continuous_pitch_feats)-window_size):
        avging_window = continuous_pitch_feats[idx:idx+window_size]
        voiced_window = avging_window!=0

        if sum(voiced_window) != irrelenvant_ind:
            window_average = round(np.average(avging_window[voiced_window]))
        else:
            window_average = 0

        average_pitches.append(window_average)

    average_pitches = np.asarray(average_pitches)
    average_pitches = np.concatenate((average_pitches, np.zeros((window_size))))
    return average_pitches


# find continuous chunks that are within tolerance of reference pitch, return the index of the random one
def best_pitch_matching_idx(average_trg_pitches, ref_pitch, tolerance=2, min_avg_pitch_dur=10):

    above_lower = average_trg_pitches > (ref_pitch - tolerance)
    below_upper = average_trg_pitches < (ref_pitch + tolerance)
    within_range_pitches = above_lower & below_upper
    eligible_run_indices = []
    vals, starts, lengths = find_runs(within_range_pitches)
    # for chunks of True in boolean array, if length is long enough, save in list
    for i in range(len(vals)):
        if vals[i]:
            if lengths[i] >= min_avg_pitch_dur:
                eligible_run_indices.append(i)
    if len(eligible_run_indices) == 0:
        return -1
    else:
        chosen_run_idx = random.choice(eligible_run_indices)
        return starts[i]





