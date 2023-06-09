import pickle
import torch
import yaml
import importlib
import random
import numpy as np
from numpy.linalg import norm

import sys, os
if os.path.abspath('../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('../my_utils'))
from my_arrays import tensor_to_array, fix_feat_length, container_to_tensor
from my_audio.pitch import midi_as_onehot

from .. import convert_utils
from .. import utils

gend_dict = {0:'M', 1:'F'}

def get_song_path(gender, gender_separated_lists, SVC_data_dir):
    
    gender_list = gender_separated_lists[gender]
    rand_int = random.randint(0,len(gender_list)-1)
    name = gender_list[rand_int]
    song_list = os.listdir(os.path.join(SVC_data_dir, name))
    song_name = random.choice(song_list)
    song_path = os.path.join(SVC_data_dir, name, song_name)

    return song_path, rand_int


def matching_pitch_clip(trg_gender, avg_src_pitch, src_path, this_train_params, subset, gender_separated_lists, track_search_tolerance=10, voiced_percent_tolerance=0.7):
    
    matched_singer_found = False
    attempt_num = 0
    while matched_singer_found==False:
        
        trg_path, trg_rand_gend_int = get_song_path(trg_gender, gender_separated_lists, os.path.join(this_train_params.SVC_feat_dir, subset))
        if os.path.dirname(trg_path) == os.path.dirname(src_path):
            continue
    
        print(f'attempt num: {attempt_num}, candidate_song: {os.path.basename(trg_path)}')
        trg_spec_feats, trg_pitch_feats = get_feats(trg_path, this_train_params.pitch_dir, subset, this_train_params.midi_range)
        continuous_pitch_feats = np.argmax(trg_pitch_feats, axis=1)
        average_trg_pitches = convert_utils.get_relevant_avg_pitches(continuous_pitch_feats, this_train_params.window_timesteps)
        start_of_chunk_idx = convert_utils.best_pitch_matching_idx(average_trg_pitches, avg_src_pitch)
        
        if start_of_chunk_idx >= 0:
            trg_pitch_clip, _ = fix_feat_length(trg_pitch_feats, this_train_params.window_timesteps, offset=start_of_chunk_idx)
            voiced = np.argmax(trg_pitch_clip, axis=1) != 0
            if (sum(voiced) / len(voiced)) < voiced_percent_tolerance:
                print('target audio didn\'t have enough voiced in it.')
                continue
            matched_singer_found = True
            break
        
        attempt_num += 1
        if attempt_num >= track_search_tolerance:
            raise convert_utils.NoMatchError(f'No matching pitches after searching {attempt_num} target candidates' )
    
    trg_spec_clip, _ = fix_feat_length(trg_spec_feats, this_train_params.window_timesteps, offset=start_of_chunk_idx)
    
    return trg_spec_clip, trg_pitch_clip, trg_rand_gend_int, start_of_chunk_idx, trg_path


def get_feats(path, pitch_dir, subset, midi_range):
    spec_feats = np.load(path)
    fn = os.path.basename(path)
    world_feats = np.load(os.path.join(pitch_dir, subset, fn.split('_')[0], fn))
    pitches = world_feats[:,-2:]
    midi_contour = pitches[:,0]
    unvoiced = pitches[:,1].astype(int) == 1
    midi_contour[unvoiced] = 0
    pitch_feats = midi_contour
    pitch_feats = midi_as_onehot(pitch_feats, midi_range)
    return spec_feats, pitch_feats


def pitch_matched_src_trg(src_gender, trg_gender, this_train_params, gender_separated_lists, subset, voiced_percent_tolerance=0.6):

    matching_target_found = False
    while not matching_target_found:
        
        src_path, src_rand_gend_int = get_song_path(src_gender, gender_separated_lists, os.path.join(this_train_params.SVC_feat_dir, subset))
        src_spec_feats, src_pitch_feats = get_feats(src_path, this_train_params.pitch_dir, subset, this_train_params.midi_range)
        src_rand_ts = random.randint(0, len(src_spec_feats)-this_train_params.window_timesteps-1)

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
                                                     this_train_params,
                                                     subset,
                                                     gender_separated_lists,
                                                     voiced_percent_tolerance=voiced_percent_tolerance)

            trg_spec_clip, trg_pitch_clip, trg_rand_gend_int, trg_rand_ts, trg_path = spec_pitch_gendint_randts_path
            try:
                src_lst_idx = gender_separated_lists[src_gender].index(os.path.basename(src_path).split('_')[0])
                gender_separated_lists[src_gender].pop(src_lst_idx)
                trg_lst_idx = gender_separated_lists[trg_gender].index(os.path.basename(trg_path).split('_')[0])
                gender_separated_lists[trg_gender].pop(trg_lst_idx)
            except Exception as e:
                print(e)
            matching_target_found = True
        except convert_utils.NoMatchError as e:
            continue
    
    src_data = src_spec_clip, src_pitch_clip, src_rand_ts, src_path
    trg_data = trg_spec_clip, trg_pitch_clip, trg_rand_ts, trg_path
            
    return src_data, trg_data


def parse_data(data, subset_metadata, subset_names, device):
    clipped_spec, clipped_pitches, rand_ts, path = data
    voice_id = os.path.basename(path).split('_')[0]
    sie_emb = subset_metadata[subset_names.index(voice_id)][1]
    arr_list = [clipped_spec, clipped_pitches, sie_emb]
    tns_list = [container_to_tensor(arr, add_batch_dim=True, device=device) for arr in arr_list]
    clipped_spec, clipped_pitches, sie_emb = tns_list
    fn = os.path.basename(path)
    return clipped_spec, clipped_pitches, sie_emb, rand_ts, fn


# something that generates pitch-matched pairs from available singers (20?) in test data
def get_svc_tensors(src_gender, trg_gender, subset_metadata, gender_separated_lists):

    subset_names = [metad[0] for metad in subset_metadata]
    src_data, trg_data = pitch_matched_src_trg(src_gender, trg_gender, this_train_params, gender_separated_lists, subset, voiced_percent_tolerance)
    src_clipped_spec_tns, src_clipped_pitch_tns, src_emb_tns, src_randts, src_fn = parse_data(src_data, subset_metadata, subset_names, device)
    trg_clipped_spec_tns, trg_clipped_pitch_tns, trg_emb_tns, trg_randts, trg_fn = parse_data(trg_data, subset_metadata, subset_names, device)

    if not this_train_params.SVC_pitch_cond:
        src_clipped_pitch_tns = None
    
    return src_clipped_spec_tns, src_emb_tns, trg_emb_tns, src_clipped_pitch_tns


def make_gen_dependancies(model_name):
    
    this_svc_model_dir = os.path.join(saved_models_dir, autovc_type, model_name)
    ckpt_path = os.path.join(this_svc_model_dir, ckpt)

    # important and confusing way of reloading a params py script and its variables
    global this_train_params
    if 'this_train_params' not in globals():
        sys.path.insert(1, this_svc_model_dir)
        import this_train_params
    else:
        del sys.path[1]
        sys.path.insert(1, this_svc_model_dir)
        importlib.reload(this_train_params)
        
    if not hasattr(this_train_params, 'pkl_fn_extras'):
        this_train_params.pkl_fn_extras = ''

#     global SVC_feat_params
    with open(os.path.join(this_train_params.SVC_feat_dir, 'feat_params.yaml')) as File:
        SVC_feat_params = yaml.load(File, Loader=yaml.FullLoader)

    pitch_dim = len(this_train_params.midi_range)+1
    if not this_train_params.SVC_pitch_cond:
        pitch_dim = 0

    print(f'Loading model: {model_name}')
    G, _, _ = utils.setup_gen(this_train_params.dim_neck,
                              this_train_params.dim_emb,
                              this_train_params.dim_pre,
                              this_train_params.sample_freq,
                              80,
                              pitch_dim,
                              device,
                              ckpt_path,
                              this_train_params.adam_init)
    G.eval()
    
    gender_separated_lists = convert_utils.get_gender_lists(os.path.join(this_train_params.SVC_feat_dir, subset))

    sie_model_name = os.path.basename(this_train_params.SIE_model_path)
    SIE_dataset_name = os.path.basename(this_train_params.SIE_feat_dir)
    metadata_path = os.path.join(metadata_root_dir,
                                 sie_model_name,
                                 SIE_dataset_name,
                                 subset,
                                 f'voices_metadata{this_train_params.pkl_fn_extras}.pkl')
    subset_metadata = pickle.load(open(metadata_path, "rb"))
    
    SIE, _ = utils.setup_sie(device, loss_device, SIE_path, this_train_params.adam_init)
    
    return G, SIE, subset_metadata, gender_separated_lists


# sie_model_name = 'autoVc_pretrainedOnVctk_Mels80'
sie_model_name = 'bestPerformingSIE_mel80'
testing_with = 'emb'
extra_note = ''
use_avg_embs = True
eval_conv = True
num_listening_studies = 23

test_on = 'damp'
ds_size = 0.0078
chosen_class_num = 100 # if this parameter is a number, this overwrites the functionality of ds_size as it specifies the number of singers to use from the subset.
ex_per_spkr = 2
num_epochs = 600
if use_avg_embs: emb_type = 'avg_'
else: emb_type = 'live_'

vc_verion = 'autoSvc'
SVC_feat_dir = f'/import/c4dm-02/bdoc3/spmel/{test_on}_qianParams'
SIE_feat_dir = f'/import/c4dm-02/bdoc3/spmel/{test_on}_qianParams'
which_cuda = 1 #when you change this, make sure synthesis.py's device is set to same
subset = 'test'
device = f'cuda:{which_cuda}'
loss_device = torch.device("cpu")
SVC_pitch_cond = False

log_step = 10
sie_emb_size = 256
voiced_percent_tolerance = 0.6
gender_conds = [(i,j) for i in range(2) for j in range(2)]

with open(os.path.join(SVC_feat_dir, 'feat_params.yaml')) as File:
    SVC_feats_params = yaml.load(File, Loader=yaml.FullLoader)
with open(os.path.join(SIE_feat_dir, 'feat_params.yaml')) as File:
    SIE_feats_params = yaml.load(File, Loader=yaml.FullLoader)

sie_dir = '/homes/bdoc3/singer-identity-encoder'
csv_dir = '/homes/bdoc3/my_data/text_data'
saved_models_dir = '/homes/bdoc3/my_data/autovc_models/'
autovc_type = 'autoSvc'
metadata_root_dir = '/homes/bdoc3/my_data/voice_embs_visuals_metadata'

ckpt = 'ckpt_500000.pt'
SIE_dataset_name = os.path.basename(SIE_feat_dir)
# SVC_dataset_name = os.path.basename(SVC_feat_dir)
# this_svc_model_dir = os.path.join(saved_models_dir, os.path.basename(vc_verion), svc_model_name)
# checkpoint_path = os.path.join(this_svc_model_dir, ckpt)
SIE_path =  os.path.join(saved_models_dir, os.path.basename(sie_dir), sie_model_name)

# gend_dict = {0:'M', 1:'F'}

total_pred_loss = 0
total_acc = 0

if eval_conv:
    assert testing_with == 'emb'
    
model_names = {'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-':'M---Sng',
               'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2':'M-E-Sng',
               'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2':'M-C-Sng',
               'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80-':'M-C-Spk'}

        
# needs a classifier model thats pretrained on test subset
results_condition = []
for i in range(num_listening_studies):
    for j, model_name in enumerate(model_names.keys()):

        # make your model here with
        
        G, SIE, subset_metadata, gender_separated_lists = make_gen_dependancies(model_name)
#         recursive_file_retrieval()

        for k, (src_gender, trg_gender) in enumerate(gender_conds):

            print(f'Getting feats for condition: {i, j, k}')
            src_clipped_spec_tns, src_emb_tns, trg_emb_tns, src_clipped_pitch_tns = get_svc_tensors(src_gender, trg_gender, subset_metadata, gender_separated_lists)
            _, converted_feats, _, _, _ = G(src_clipped_spec_tns, src_emb_tns, trg_emb_tns, src_clipped_pitch_tns)
            converted_sie_emb = tensor_to_array(SIE(converted_feats.squeeze(1)))
            trg_emb_arr = tensor_to_array(trg_emb_tns)
            cosine_sim = np.dot(converted_sie_emb, trg_emb_arr)/(norm(converted_sie_emb) * norm(trg_emb_arr))
            print(cosine_sim)
            if cosine_sim > 1:
                raise Exception
            euclidean_distance = np.linalg.norm(converted_sie_emb - trg_emb_arr)
            results_condition.append((cosine_sim, euclidean_distance, (j, model_name), (k, (src_gender, trg_gender))))

# with open('conversion_cosine_results.pkl', 'wb') as handle:
with open('conversion_cosine_results.pkl', 'wb') as handle:
    pickle.dump(results_condition, handle, protocol=pickle.HIGHEST_PROTOCOL)