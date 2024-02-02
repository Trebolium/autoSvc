import random
import os
import sys
import csv
import pdb
import numpy as np
this_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(this_script_dir)
super_dir = os.path.dirname(root_dir)
my_utils_dir = os.path.join(super_dir, 'my_utils')
if os.path.abspath(my_utils_dir) not in sys.path: sys.path.insert(1, os.path.abspath(my_utils_dir))
from my_arrays import find_runs
from my_os import recursive_file_retrieval


def get_damp_gender(
    ignore_unknowns=False,
    csv_path="",
):
    """
    Get entries from gender csv file, return single list of performer-gender tuples
    """

    f = open(csv_path, "r")
    reader = csv.reader(f)
    header = next(reader)
    singer_meta = [row for row in reader]
    if ignore_unknowns:
        performer_gender_list = [
            (row[0].split("_")[0], row[8]) for row in singer_meta if row[8] != " None"
        ]
    else:
        performer_gender_list = [(row[0].split("_")[0], row[8])
                                 for row in singer_meta]

    return performer_gender_list


def get_gender_lists(SVC_data_dir, csv_path):
    print("Getting gender info...")
    performer_gender_list = get_damp_gender(ignore_unknowns=True, csv_path=csv_path)
    _, spmel_fps = recursive_file_retrieval(SVC_data_dir)
    spmel_perf_id = [os.path.basename(fp).split("_")[0] for fp in spmel_fps]
    performance_gender_subset = [
        perf_gen for perf_gen in performer_gender_list if perf_gen[0] in spmel_perf_id
    ]
    females = [perf for perf, gend in performance_gender_subset if gend == " F"]
    males = [perf for perf, gend in performance_gender_subset if gend == " M"]
    gender_separated_lists = [males, females]
    return gender_separated_lists


class NoMatchError(Exception):
    pass


def get_relevant_avg_pitches(continuous_pitch_feats, window_size, irrelenvant_ind=0):
    average_pitches = []
    for idx in range(len(continuous_pitch_feats) - window_size):
        avging_window = continuous_pitch_feats[idx : idx + window_size]
        voiced_window = avging_window != 0

        if sum(voiced_window) != irrelenvant_ind:
            window_average = round(np.average(avging_window[voiced_window]))
        else:
            window_average = 0

        average_pitches.append(window_average)

    average_pitches = np.asarray(average_pitches)
    average_pitches = np.concatenate((average_pitches, np.zeros((window_size))))
    return average_pitches


# find continuous chunks that are within tolerance of reference pitch, return the index of the random one
def best_pitch_matching_idx(
    average_trg_pitches, ref_pitch, tolerance=2, min_avg_pitch_dur=10
):
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
