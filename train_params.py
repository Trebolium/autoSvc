
# paths, dirs, trg ds
svc_model_name = 'defaultName'
svc_model_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
SIE_feat_dir = '/homes/bdoc3/my_data/world_vocoder_data/m4a2worldChandna'
# fill this out only if you want SVC input features to be different to SIE input features
SVC_feat_dir = '/homes/bdoc3/my_data/world_vocoder_data/m4a2worldChandna'
use_loader = 'vocal'

# evaluation feature paths
medley_data_path = '/homes/bdoc3/my_data/spmel_data/medley/singer_chunks_metadata.pkl'
vocalset_data_path = '/homes/bdoc3/my_data/phonDet/spmel_autovc_params_normalized'
vctk_data_path = '/homes/bdoc3/my_data/spmel_data/medley/all_meta_data.pkl'

# Singer identity encoder
SIE_path = '/homes/bdoc3/singer-identity-encoder/encoder/saved_models/dampInton500Voices_chandnaProcessing_unnormed_ge2e'

# autoSvc init
which_cuda = 0
adam_init = 0.0001
batch_size = 2
svc_ckpt_path = ''
sample_freq = 16
dim_emb = 256
dim_neck = 32
dim_pre = 512
pitch_cond = False
use_ckpt_config = False

# autoSvc training
eval_all = False
prnt_loss_weight = 1.0
psnt_loss_weight = 1.0
which_embs = 'sie-live'

# data details
chunk_num = 6
chunk_seconds = 0.5
window_timesteps = 192
use_aper_feats = False
midi_range = range(0, 100)

# schedulers
ckpt_freq = 50000
log_step = 10
max_iters = 1000000
patience = 30
spec_freq = 10000
train_iter = 500
train_size = 20

# model params
model_hidden_size = 256
model_embedding_size = 256
model_num_layers = 3