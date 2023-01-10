import os
import pickle
import torch
import yaml
import importlib
import sys
import pdb
import numpy as np
from numpy.linalg import norm
sys.path.insert(1, '/homes/bdoc3/my_utils')
import utils
from my_arrays import tensor_to_array
from convert_utils import get_gender_lists, pitch_matched_src_trg, parse_data


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
    
    gender_separated_lists = get_gender_lists(os.path.join(this_train_params.SVC_feat_dir, subset))

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
#             pdb.set_trace()
            converted_sie_emb = tensor_to_array(SIE(converted_feats.squeeze(1)))
            trg_emb_arr = tensor_to_array(trg_emb_tns)
            # sims = []
            # dists = []
            # for l in range(len(subset_metadata)):
            #     entry = subset_metadata[l]
            #     s_id = entry[0]
            #     s_emb = entry[2]
            cosine_sim = np.dot(converted_sie_emb, trg_emb_arr)/(norm(converted_sie_emb) * norm(trg_emb_arr))
            print(cosine_sim)
            if cosine_sim > 1:
                pdb.set_trace()
            euclidean_distance = np.linalg.norm(converted_sie_emb - trg_emb_arr)
            results_condition.append((cosine_sim, euclidean_distance, (j, model_name), (k, (src_gender, trg_gender))))

# with open('conversion_cosine_results.pkl', 'wb') as handle:
with open('deletable.pkl', 'wb') as handle:
    pickle.dump(results_condition, handle, protocol=pickle.HIGHEST_PROTOCOL)