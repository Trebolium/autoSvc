import os
import pickle
import torch
import time
import datetime
import torch.nn.functional as F
import yaml
import sys
import pdb
import numpy as np
from numpy.linalg import norm
from model_vc import Aux_Voice_Classifier
from torch.utils.tensorboard import SummaryWriter
from data_objects.data_loaders import load_primary_dataloader
sys.path.insert(1, '/homes/bdoc3/my_utils')
import utils
from neural.eval import get_accuracy
from my_arrays import fix_feat_length, container_to_tensor, tensor_to_array, find_runs
from my_os import recursive_file_retrieval
from my_audio.pitch import midi_as_onehot
from convert_params import get_gender_lists, pitch_matched_src_trg, parse_data, get_fn_string


# svc_model_name = 'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80-'
# svc_model_name = 'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-'
# svc_model_name = 'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2'
svc_model_name = 'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2'

# sie_model_name = 'autoVc_pretrainedOnVctk_Mels80'
sie_model_name = 'bestPerformingSIE_mel80'
testing_with = 'emb'
extra_note = ''
use_avg_embs = True
eval_conv = True

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
SVC_pitch_cond = False

num_listening_studies = 23
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
metadata_root_dir = '/homes/bdoc3/my_data/voice_embs_visuals_metadata'

SIE_dataset_name = os.path.basename(SIE_feat_dir)
# SVC_dataset_name = os.path.basename(SVC_feat_dir)
this_svc_model_dir = os.path.join(saved_models_dir, os.path.basename(vc_verion), svc_model_name)
checkpoint_path = os.path.join(this_svc_model_dir, 'saved_model.pt')
SIE_path =  os.path.join(saved_models_dir, os.path.basename(sie_dir), sie_model_name)

# gend_dict = {0:'M', 1:'F'}

total_pred_loss = 0
total_acc = 0

if eval_conv:
    assert testing_with == 'emb'

### Make model

sys.path.insert(1, this_svc_model_dir)
import this_train_params
if not hasattr(this_train_params, 'pkl_fn_extras'):
    this_train_params.pkl_fn_extras = ''

G, _, _ = utils.setup_gen(this_train_params.dim_neck,
                            this_train_params.dim_emb,
                            this_train_params.dim_pre,
                            this_train_params.sample_freq,
                            80, 0, device, checkpoint_path, this_train_params.adam_init)
G.eval()

gender_separated_lists = get_gender_lists(this_train_params.SVC_data_dir)

if use_avg_embs:
    metadata_path = os.path.join(metadata_root_dir, sie_model_name, SIE_dataset_name, subset, f'voices_metadata{this_train_params.pkl_fn_extras}.pkl')
    subset_metadata = pickle.load(open(metadata_path, "rb"))
    subset_names = [metad[0] for metad in subset_metadata]
else:
    loss_device = torch.device("cpu")
    SIE, _ = utils.setup_sie(device, loss_device, SIE_path, this_train_params.adam_init)

start_time = time.time()

model_names = {'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-':'M---Sng',
               'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2':'M-E-Sng',
               'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80-':'M-C-Spk',
               'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2':'M-C-Sng'}

# something that generates pitch-matched pairs from available singers (20?) in test data

def get_svc_tensors(src_gender, trg_gender):

    src_data, trg_data = pitch_matched_src_trg(src_gender, trg_gender, this_train_params, gender_separated_lists, voiced_percent_tolerance)
    src_clipped_spec_tns, src_clipped_pitch_tns, src_emb_tns, src_randts, src_fn = parse_data(src_data, subset_metadata, subset_names, device)
    trg_clipped_spec_tns, trg_clipped_pitch_tns, trg_emb_tns, trg_randts, trg_fn = parse_data(trg_data, subset_metadata, subset_names, device)

    if not this_train_params.SVC_pitch_cond:
        src_clipped_pitch_tns = None
    
    return src_clipped_spec_tns, src_emb_tns, trg_emb_tns, src_clipped_pitch_tns


# needs a classifier model thats pretrained on test subset
results_condition = []

for i in range(num_listening_studies):
    for j, model_name in enumerate(model_names.keys()):

        for k, (src_gender, trg_gender) in enumerate(gender_conds):

            print(f'Getting feats for condition: {i, j, k}')
            src_clipped_spec_tns, src_emb_tns, trg_emb_tns, src_clipped_pitch_tns = get_svc_tensors(src_gender, trg_gender)
            _, converted_feats, _, _, _ = G(src_clipped_spec_tns, src_emb_tns, trg_emb_tns, src_clipped_pitch_tns)
            converted_sie_emb = SIE(converted_feats)
            cosine_sim = np.dot(converted_sie_emb,trg_emb_tns)/(norm(converted_sie_emb)*norm(trg_emb_tns))
            results_condition.append((cosine_sim, (j, model_name), (k, (src_gender, trg_gender))))

with open('conversion_cosine_results.pkl', 'wb') as handle:
    pickle.dump(results_condition, handle, protocol=pickle.HIGHEST_PROTOCOL)