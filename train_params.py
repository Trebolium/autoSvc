import os
import sys

this_script_dir = os.path.dirname(os.path.abspath(__file__))
super_dir = os.path.dirname(this_script_dir)
my_utils_dir = os.path.join(super_dir, 'my_utils')
sie_dir = os.path.join(super_dir, 'singer-identity-encoder')

which_cuda = 1

use_loader = "damp" # name of the dataset being used
svc_feat_type = "mel" # name of feature type (either mel or world)
sie_feat_type = "mel"
subset_size = 1  # should present a fraction of the dataset
chosen_class_num = None

# AUTOSVC TRAINING

# aper = aperiodicity features taken from WORLD vocoder output
use_aper_feats = False
if use_aper_feats:
    apers_string = "WithApers"
else:
    apers_string = ""
ds_fn = f"{use_loader}_{svc_feat_type}{apers_string}_Size{str(subset_size)}"

# boolean to determine whether a lookup table is used for SIE-generation, or whether they are generated live during training
use_avg_singer_embs = True
if use_avg_singer_embs:
    embs_string = "avgEmbs"
else:
    embs_string = "liveEmbs"

# losses to include
include_code_loss = False
use_emb_loss = True
use_aux_classer = False

if include_code_loss:
    ccloss_string = "CcLoss"
else:
    ccloss_string = ""
if use_emb_loss:
    embloss_string = "EmbLoss"
else:
    embloss_string = ""
if use_aux_classer:
    embloss_string = "ClassLoss"
else:
    classloss_string = ""

loss_type = "L2loss"

# Pitch-related

pitch_dir = "damp_example_pitch"
SIE_pitch_cond = False
if SIE_pitch_cond:
    siePitchCond = "siePitchCond"
else:
    siePitchCond = ""
SVC_pitch_cond = False
if SVC_pitch_cond:
    svcPitchCond = "svcPitchCond"
else:
    svcPitchCond = ""

# Normalisation methods

norm_method = None
if norm_method != None:
    norm_method_string = norm_method
else:
    norm_method_string = ""

# Earlystopping params

patience = 20  # choose 20 when considering damp dataset
patience_thresh = 0.0  # 0.1 for damp dataset
early_stopping_loss = "total"  # 'recon', 'cc', 'sie', 'total'

# SIE model and lookup table info

training_fn = f"{embs_string}_{ccloss_string}{embloss_string}{classloss_string}{loss_type}_{siePitchCond}{svcPitchCond}_{norm_method_string}"

# SIE_model_path = os.path.join(super_dir, 'my_data/autovc_models/singer-identity-encoder/bestPerformingSIE_mel80')
SIE_model_path = os.path.join(sie_dir, 'sie_models', 'default_model')
qians_pretrained_model = False

pkl_fn_extras = "_100avg"
fn_final_notes = "deletable"

# SVC model name

SVC_model_name = (
    f"{ds_fn}-{training_fn}-{os.path.basename(SIE_model_path)}-{fn_final_notes}"
)
print("Model name: ", SVC_model_name)

SVC_models_dir = os.path.join(this_script_dir, 'models')
svc_ckpt_path = ""
SVC_feat_dir = os.path.join(sie_dir, 'damp_example_feats')
SIE_feat_dir = SVC_feat_dir
metadata_path = os.path.join(super_dir, 'voice_embs_visuals_metadata')

log_step = 10
max_iters = 50  # measured in batches, not data examples
max_cycle_iters = 50
ckpt_freq = 50000  # make this number higher than max_iters to avoid ckpts altogether
spec_freq = 50000

use_ckpt_config = False
# SVC params
adam_init = 0.0001
batch_size = 2  # 32bs means 88 datas per train epoch, 11 per val epoch
sample_freq = (
    32  # freq by which we sample the codes after encoder, NOT samplerate - usually 16
)
dim_emb = 256 # dimensionality of sie embeddings
dim_neck = 32  # dimensionality of 
dim_pre = 512 # dimensionality of conv channels
num_workers = 0  # match this to batch size
prnt_loss_weight = 1.0
psnt_loss_weight = 1.0
code_loss_weight = 1.0
# SIE model params
model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3

# data details
window_timesteps = (
    128  # Window size does not affect accuracy, just speed of network to converge
)
midi_range = range(
    36, 82
)  # should be the same as used in SIE if trained with pitch conditioning as well
