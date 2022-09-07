import os
#LAST CHANGED
# batch_size 
# num_workers 
# svc_model_name 
# which_cuda
# use_loader
# use_aper_feats
# SIE_pitch_cond, SVC_pitch_cond
# use_avg_singer_embs
# include_code_loss

# paths, dirs, trg ds
pitch_dir = '/import/c4dm-02/bdoc3/world_data/damp_80_16ms' #it doesn't matter what the dims are, this data stays the same I think
# pitch_dir = '/homes/bdoc3/my_data/world_vocoder_data/vctk_new_ref'

svc_model_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
# svc_model_name = 'dampWorld60Conversion_withBestPerformingSIE_vanillaSetup_cont'
# svc_model_name = 'dampWorld80Conversion_withBestPerformingSIE_pitchCond'
svc_model_name = 'deletable'

use_loader = 'damp'

# Singer identity encoder - DO not change for DAMP anymore
SIE_path = '/homes/bdoc3/my_data/autovc_models/singer_identity_encoder/bestPerformingSIE_mel80'
SIE_feat_dir = '/import/c4dm-02/bdoc3/spmel/damp_qianParams' #always MEL-type if using bestPerformingSIE

# fill this out only if you want SVC input features to be different to SIE input features
# SVC_feat_dir = '/homes/bdoc3/my_data/world_vocoder_data/damp_new'
SVC_feat_dir = '/import/c4dm-02/bdoc3/world_data/damp_60_16ms'
# SVC_feat_dir = '/import/c4dm-02/bdoc3/spmel/damp_qianParams'


# autoSvc init
which_cuda = 1
adam_init = 0.0001
batch_size = 2 # 32bs means 88 datas per train epoch, 11 per val epoch
svc_ckpt_path = os.path.join(svc_model_dir, 'dampWorld60Conversion_withBestPerformingSIE_vanillaSetup', 'saved_model.pt')
# svc_ckpt_path = ''
sample_freq = 16 #freq by which we sample the codes after encoder, NOT samplerate - usually 16
dim_emb = 256
dim_neck = 32 #32
dim_pre = 512
SIE_pitch_cond = False
SVC_pitch_cond = False
use_ckpt_config = False
num_workers = 2 # match this to batch size

# autoSvc training
eval_all = False
# max_cycle_iters = 500 # how frequently do we wanna check in with the vals?
prnt_loss_weight = 1.0
psnt_loss_weight = 1.0
code_loss_weight = 1.0
use_avg_singer_embs = True

# data details
chunk_num = 6
chunk_seconds = 0.5
window_timesteps = 128 # Window size does not affect accuracy, just speed of network to converge
use_aper_feats = False
midi_range = range(36, 82) # should be the same as used in SIE if trained with pitch conditioning as well
norm_method = None

# schedulers
ckpt_freq = 50000
log_step = 10
max_iters = 1000000 #measured in batches, not data examples
max_cycle_iters = 1000
patience = 30
spec_freq = 500

# SIE model params
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3

include_code_loss = False