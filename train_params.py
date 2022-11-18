import os

which_cuda = 2

use_loader = 'damp'
feat_type = 'world'
subset_size = 0.25 # should present a fraction of the dataset
use_aper_feats = True
if use_aper_feats: apers_string = 'WithApers'
else: apers_string = ''
ds_fn = f'{use_loader}_{feat_type}{apers_string}_Size{str(subset_size)}'

# autoSvc training
use_avg_singer_embs = True
if use_avg_singer_embs: embs_string = 'avgEmbs'
else: embs_string = 'liveEmbs'


include_code_loss = False # dsicvoered this is necessary for good vc, so leave it in from now on
if include_code_loss: ccloss_string = 'CcLoss'
else: ccloss_string = ''
use_emb_loss = False
if use_emb_loss: embloss_string = 'EmbLoss'
else: embloss_string = ''
use_aux_classer = False
if use_aux_classer: embloss_string = 'ClassLoss'
else: classloss_string = ''


pitch_dir = '/import/c4dm-02/bdoc3/world_data/damp_80_16ms' #it doesn't matter what the dims are, this data stays the same I think
# pitch_dir = '/homes/bdoc3/my_data/world_vocoder_data/vctk_new_ref'
SIE_pitch_cond = False
if SIE_pitch_cond: siePitchCond = 'siePitchCond'
else: siePitchCond = ''
SVC_pitch_cond = True
if SVC_pitch_cond: svcPitchCond = 'svcPitchCond'
else: svcPitchCond = ''


norm_method = None
if norm_method != None: norm_method_string = norm_method
else: norm_method_string = ''

patience = 1000000 # choose 20 when considering damp dataset
patience_thresh = 0.0 #0.1 for damp dataset
early_stopping_loss = 'total' # 'recon', 'cc', 'sie', 'total'

training_fn = f'{embs_string}_{ccloss_string}{embloss_string}{classloss_string}_{siePitchCond}{svcPitchCond}_{norm_method_string}'


SVC_models_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
# SIE_model_path = '/homes/bdoc3/my_data/autovc_models/singer-identity-encoder/autoVc_pretrainedOnVctk_Mels80'
SIE_model_path = '/homes/bdoc3/my_data/autovc_models/singer-identity-encoder/bestPerformingSIE_mel80'
# SIE_model_path = '/homes/bdoc3/my_data/autovc_models/singer-identity-encoder/sie_VoxCeleb1LibrispeechDampQianParams_mels80'
# svc_ckpt_path = os.path.join(SVC_models_dir, 'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters', 'saved_model.pt')
svc_ckpt_path = ''
pkl_fn_extras = ''
fn_final_notes = ''

SVC_model_name = f'{ds_fn}-{training_fn}-{os.path.basename(SIE_model_path)}-{fn_final_notes}'
# SVC_model_name = 'damp_freq32QuarterDataset_10EmbAverage_patience20Thresh.1-bestPerformingSIE_mel80'
print('Model name: ', SVC_model_name)

# fill this out only if you want SVC input features to be different to SIE input features
SVC_feat_dir = '/import/c4dm-02/bdoc3/world_data/damp_80_16ms'
# SVC_feat_dir = f'/import/c4dm-02/bdoc3/spmel/{use_loader}_qianParams'
SIE_feat_dir = f'/import/c4dm-02/bdoc3/spmel/{use_loader}_qianParams' #always MEL-type if using bestPerformingSIE
# SVC_feat_dir = '/homes/bdoc3/my_data/spmel_data/vocadito'
# SIE_feat_dir = SVC_feat_dir

# schedulers
# ckpt_freq = 1000
log_step = 10
max_iters = 500001 #measured in batches, not data examples
max_cycle_iters = 1000
ckpt_freq = 500000 #make this number higher than max_iters to avoid ckpts altogether
spec_freq = 50000

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
chunk_num = 6
chunk_seconds = 0.5
window_timesteps = 128 # Window size does not affect accuracy, just speed of network to converge
midi_range = range(36, 82) # should be the same as used in SIE if trained with pitch conditioning as well





