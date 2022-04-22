
# paths, dirs, trg ds
svc_model_name = 'defaultName'
model_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
feature_dir = '/homes/bdoc3/my_data/spmel_data/damp_inton/binNormmed_split_by_voice'
use_loader = 'vocal'

# Singer identity encoder
SIE_path = '/homes/bdoc3/singer-identity-encoder/encoder/saved_models/dampInton500Voices_chandnaProcessing_unnormed_ge2e'
ckpt_iters = 300000

# autoSvc init
which_cuda = 0
adam_init = 0.0001
batch_size = 2
svc_ckpt_path = ''
freq = 16
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
autosvc_crop = 160

# schedulers
ckpt_freq = 50000
log_step = 10
max_iters = 300000
patience = 501
spec_freq = 10000
train_iter = 500
train_size = 20