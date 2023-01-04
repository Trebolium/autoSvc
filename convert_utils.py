import random, os, pdb, sys
import numpy as np
import utils

sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_arrays import fix_feat_length, container_to_tensor, tensor_to_array, find_runs
from my_os import recursive_file_retrieval
from my_audio.pitch import midi_as_onehot


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

gend_dict = {0:'M', 1:'F'}

class NoMatchError(Exception):
    pass

def parse_data(data, subset_metadata, subset_names, device):
    clipped_spec, clipped_pitches, rand_ts, path = data
    voice_id = os.path.basename(path).split('_')[0]
    sie_emb = subset_metadata[subset_names.index(voice_id)][1]
    arr_list = [clipped_spec, clipped_pitches, sie_emb]
    tns_list = [container_to_tensor(arr, add_batch_dim=True, device=device) for arr in arr_list]
    clipped_spec, clipped_pitches, sie_emb = tns_list
    fn = os.path.basename(path)
    return clipped_spec, clipped_pitches, sie_emb, rand_ts, fn


def get_fn_string(src_fn, trg_fn, src_rand_ts, model_str, gen_pair):
    src_gender, trg_gender = gen_pair
    src_str = gend_dict[src_gender] +src_fn.split('_')[0]
    trg_str = gend_dict[trg_gender] +trg_fn.split('_')[0]
    return model_str +f'_{src_str}' +f'_timestep{src_rand_ts}' +f'_{trg_str}'


def pitch_matched_src_trg(src_gender, trg_gender, this_train_params, gender_separated_lists, voiced_percent_tolerance=0.6):

    matching_target_found = False
    while not matching_target_found:
        src_path, src_rand_gend_int = get_song_path(src_gender, gender_separated_lists, this_train_params.SVC_data_dir)
    #     src_path = '/import/c4dm-02/bdoc3/spmel/damp_qianParams/test/434587164/434587164_2141814685.npy'

        src_spec_feats, src_pitch_feats = get_feats(src_path)

        src_rand_ts = random.randint(0, len(src_spec_feats)-this_train_params.window_timesteps-1)
    #     src_rand_ts = 2981

        src_spec_clip, _ = fix_feat_length(src_spec_feats, this_train_params.window_timesteps, offset=src_rand_ts)
        src_pitch_clip, _ = fix_feat_length(src_pitch_feats, this_train_params.window_timesteps, offset=src_rand_ts)
        # ensure we do not include avereaging over zero values which represents unvoiced
        voiced = np.argmax(src_pitch_clip,  axis=1)!=0
        if (sum(voiced) / len(voiced)) < voiced_percent_tolerance:
            continue
        avg_src_pitch = round(np.average(np.argmax(src_pitch_clip, axis=1)[voiced]))

        print(f'src_song: {os.path.basename(src_path)}, rand_int: {src_rand_ts}, src_gend: {gend_dict[src_gender]}, avg_src_pitch: {avg_src_pitch}')

        print(avg_src_pitch)
        try:
            spec_pitch_gendint_randts_path = matching_pitch_clip(trg_gender,
                                                     avg_src_pitch,
                                                     src_path,
                                                     voiced_percent_tolerance=voiced_percent_tolerance)

            trg_spec_clip, trg_pitch_clip, trg_rand_gend_int, trg_rand_ts, trg_path = spec_pitch_gendint_randts_path
            try:
                src_lst_idx = gender_separated_lists[src_gender].index(os.path.basename(src_path).split('_')[0])
                gender_separated_lists[src_gender].pop(src_lst_idx)
                trg_lst_idx = gender_separated_lists[trg_gender].index(os.path.basename(trg_path).split('_')[0])
                gender_separated_lists[trg_gender].pop(trg_lst_idx)
            except Exception as e:
                print(e)
                pdb.set_trace() 
            matching_target_found = True
        except NoMatchError as e:
            continue
    
    src_data = src_spec_clip, src_pitch_clip, src_rand_ts, src_path
    trg_data = trg_spec_clip, trg_pitch_clip, trg_rand_ts, trg_path
            
    return src_data, trg_data


def get_song_path(gender, gender_separated_lists, SVC_data_dir):
    
    gender_list = gender_separated_lists[gender]
    rand_int = random.randint(0,len(gender_list)-1)
    name = gender_list[rand_int]
    song_list = os.listdir(os.path.join(SVC_data_dir, name))
    song_name = random.choice(song_list)
    song_path = os.path.join(SVC_data_dir, name, song_name)

    return song_path, rand_int


def get_feats(path):
    spec_feats = np.load(path)
    fn = os.path.basename(path)
    world_feats = np.load(os.path.join(pitch_dir, subset, fn.split('_')[0], fn))
    pitches = world_feats[:,-2:]
    midi_contour = pitches[:,0]
    unvoiced = pitches[:,1].astype(int) == 1
    midi_contour[unvoiced] = 0
    pitch_feats = midi_contour
    pitch_feats = midi_as_onehot(pitch_feats, this_train_params.midi_range)
    return spec_feats, pitch_feats




def matching_pitch_clip(trg_gender, avg_src_pitch, src_path, track_search_tolerance=10, voiced_percent_tolerance=0.7):
    
    matched_singer_found = False
    attempt_num = 0
    while matched_singer_found==False:
        
        trg_path, trg_rand_gend_int = get_song_path(trg_gender)
        if os.path.dirname(trg_path) == os.path.dirname(src_path):
            continue
    
        print(f'attempt num: {attempt_num}, candidate_song: {os.path.basename(trg_path)}')
        trg_spec_feats, trg_pitch_feats = get_feats(trg_path)
        continuous_pitch_feats = np.argmax(trg_pitch_feats, axis=1)
        average_trg_pitches = get_relevant_avg_pitches(continuous_pitch_feats, this_train_params.window_timesteps)
        start_of_chunk_idx = best_pitch_matching_idx(average_trg_pitches, avg_src_pitch)
        
        if start_of_chunk_idx >= 0:
            trg_pitch_clip, _ = fix_feat_length(trg_pitch_feats, this_train_params.window_timesteps, offset=start_of_chunk_idx)
            voiced = np.argmax(trg_pitch_clip, axis=1) != 0
            if (sum(voiced) / len(voiced)) < voiced_percent_tolerance:
                continue
            matched_singer_found = True
            break
        
        attempt_num += 1
        if attempt_num >= track_search_tolerance:
            raise NoMatchError(f'No matching pitches after searching {attempt_num} target candidates' )
            
    trg_spec_clip, _ = fix_feat_length(trg_spec_feats, this_train_params.window_timesteps, offset=start_of_chunk_idx)
    
    return trg_spec_clip, trg_pitch_clip, trg_rand_gend_int, start_of_chunk_idx, trg_path


