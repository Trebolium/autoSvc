import os, pdb

which_cuda = 2

use_loader = 'damp'
svc_feat_type = 'mel'
sie_feat_type = 'mel'
subset_size = 1 # should present a fraction of the dataset
chosen_class_num = None

# AUTOSVC TRAINING

use_aper_feats = False
if use_aper_feats: apers_string = 'WithApers'
else: apers_string = ''
ds_fn = f'{use_loader}_{svc_feat_type}{apers_string}_Size{str(subset_size)}'

use_avg_singer_embs = True
if use_avg_singer_embs: embs_string = 'avgEmbs'
else: embs_string = 'liveEmbs'

# losses

include_code_loss = False # dsicvoered this is necessary for good vc, so leave it in from now on
if include_code_loss: ccloss_string = 'CcLoss'
else: ccloss_string = ''

use_emb_loss = False
if use_emb_loss: embloss_string = 'EmbLoss'
else: embloss_string = ''

use_aux_classer = False
if use_aux_classer: embloss_string = 'ClassLoss'
else: classloss_string = ''

loss_type = 'L1loss'

# Pitch-related

pitch_dir = 'example_crepe'
# pitch_dir = '/homes/bdoc3/my_data/crepe_data/damp'
SIE_pitch_cond = False
if SIE_pitch_cond: siePitchCond = 'siePitchCond'
else: siePitchCond = ''
SVC_pitch_cond = False
if SVC_pitch_cond: svcPitchCond = 'svcPitchCond'
else: svcPitchCond = ''

# Normalisation methods
 
norm_method = None
if norm_method != None: norm_method_string = norm_method
else: norm_method_string = ''

# Earlystopping params

patience = 500000 # choose 20 when considering damp dataset
patience_thresh = 0.0 #0.1 for damp dataset
early_stopping_loss = 'total' # 'recon', 'cc', 'sie', 'total'

training_fn = f'{embs_string}_{ccloss_string}{embloss_string}{classloss_string}{loss_type}_{siePitchCond}{svcPitchCond}_{norm_method_string}'



# SIE_model_path = '/homes/bdoc3/my_data/autovc_models/singer-identity-encoder/sie_vctkQianParams_mels80'
SIE_model_path = '../singer-identity-encoder/sie_models/default_model'

pkl_fn_extras = '_100avg'
fn_final_notes = ''

SVC_model_name = f'{ds_fn}-{training_fn}-{os.path.basename(SIE_model_path)}-{fn_final_notes}'
# SVC_model_name = 'defaultName'
print('Model name: ', SVC_model_name)

# SVC_models_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
SVC_models_dir = './autosvc_models'
# svc_ckpt_path = os.path.join(SVC_models_dir, SVC_model_name, 'saved_model.pt')
svc_ckpt_path = ''

# SIE_feat_dir = f'/import/c4dm-02/bdoc3/spmel/{use_loader}_qianParams'
SVC_feat_dir = f'../singer-identity-encoder/example_feats'
SIE_feat_dir = SVC_feat_dir
metadata_path = '../voice_embs_visuals_metadata'

# schedulers
# ckpt_freq = 1000
log_step = 10
max_iters = 100 #measured in batches, not data examples
max_cycle_iters = 1000
ckpt_freq = 250000 #make this number higher than max_iters to avoid ckpts altogether
spec_freq = 5000

use_ckpt_config = False
#SVC params
adam_init = 0.0001
batch_size = 2 # 32bs means 88 datas per train epoch, 11 per val epoch
sample_freq = 32 #freq by which we sample the codes after encoder, NOT samplerate - usually 16
dim_emb = 256
dim_neck = 32 #32
dim_pre = 512
num_workers = 0 # match this to batch size
prnt_loss_weight = 1.0
psnt_loss_weight = 1.0
code_loss_weight = 1.0
# SIE model params
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3

# data details
window_timesteps = 128 # Window size does not affect accuracy, just speed of network to converge
midi_range = range(36, 82) # should be the same as used in SIE if trained with pitch conditioning as well





