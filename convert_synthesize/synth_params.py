import os, sys

this_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(this_script_dir)
super_dir = os.path.dirname(root_dir)
sie_dir = os.path.join(super_dir, 'singer-identity-encoder')

svc_model_dir = "./autosvc_models"
svc_model_name = "damp_mel_Size1-avgEmbs_L1loss__-default_model-"
# svc_model_name = 'deletable'

use_loader = "damp"

# Singer identity encoder
SIE_path = ".../singer-identity-encoder/sie_models/default_model"
# SIE_feat_dir = '/import/c4dm-02/bdoc3/spmel/vctk_qianParams'
SIE_feat_dir = "../singer-identity-encoder/example_feats"

# fill this out only if you want SVC input features to be different to SIE input features
# SVC_feat_dir = '/homes/bdoc3/my_data/world_vocoider_data/damp_new'
SVC_feat_dir = os.path.join(sie_dir, 'damp_example_feats')

# data details
window_timesteps = (
    128  # Window size does not affect accuracy, just speed of network to converge
)
midi_range = range(
    36, 82
)  # should be the same as used in SIE if trained with pitch conditioning as well
