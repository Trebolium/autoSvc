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
pitch_dir = '/homes/bdoc3/my_data/world_vocoder_data/damp_new'

svc_model_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
# svc_model_name = 'mel80vctk_NOzerodPitchCond_Window384_autovcBasicLossFunction'
svc_model_name = 'withImprovedMelSIE_mel80vctk'
# svc_model_name = 'deletable'

use_loader = 'vctk'

# Singer identity encoder
# SIE_path = '/homes/bdoc3/my_data/autovc_models/singer_identity_encoder/autoVc_pretrained/continuedFromPretrained'
SIE_path = '/homes/bdoc3/my_data/autovc_models/singer_identity_encoder/vctk_correctedItersAndMels80'
SIE_feat_dir = '/import/c4dm-02/bdoc3/spmel/vctk_qianParams'
# SIE_feat_dir = '/import/c4dm-02/bdoc3/spmel/vctk80'

# fill this out only if you want SVC input features to be different to SIE input features
# SVC_feat_dir = '/homes/bdoc3/my_data/world_vocoder_data/damp_new'
# SVC_feat_dir = '/import/c4dm-02/bdoc3/world_data/damp_60'
SVC_feat_dir = '/import/c4dm-02/bdoc3/spmel/vctk_qianParams'


# autoSvc init
which_cuda = 2  
adam_init = 0.0001
batch_size = 2 # 32bs means 88 datas per train epoch, 11 per val epoch
svc_ckpt_path = ''
sample_freq = 32 #freq by which we sample the codes after encoder, NOT samplerate
dim_emb = 256
dim_neck = 2 #32
dim_pre = 512
SIE_pitch_cond = False
SVC_pitch_cond = False
use_ckpt_config = False
num_workers = 0 # match this to batch size

# autoSvc training
eval_all = False
max_cycle_iters = 500 # how frequently do we wanna check in with the vals?
prnt_loss_weight = 1.0
psnt_loss_weight = 1.0
code_loss_weight = 1.0
use_avg_singer_embs = True

# data details
chunk_num = 6
chunk_seconds = 0.5
window_timesteps = 384 # Window size does not affect accuracy, just speed of network to converge
use_aper_feats = False
midi_range = range(36, 82) # should be the same as used in SIE if trained with pitch conditioning as well
norm_method = None

# schedulers
ckpt_freq = 50000
log_step = 10
max_iters = 1000000 #measured in batches, not data examples
patience = 30
spec_freq = 50000

# SIE model params
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3

include_code_loss = False