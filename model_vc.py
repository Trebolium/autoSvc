import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_params import *


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


# "4.2. The Content Encoder"
class Encoder(nn.Module):
    """Encoder module:"""

    def __init__(self, dim_neck, dim_emb, freq, num_spec_feats):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        convolutions = []
        for i in range(3):
            # "the input to the content encoder is the 80-dimensional mel-spectrogram of X1 concatenated with the speaker embedding" - I think the embeddings are copy pasted from a dataset, as the Speaker Decoder is pretrained and may not actually appear in this implementation?
            conv_layer = nn.Sequential(
                # "the input to the content encoder is the 80-dimensional mel-spectrogram of X1 concatenated with the speaker embedding. The concatenated features are fed into three 5 × 1 convolutional layers, each followed by batch normalization and ReLU activation. The number of channels i
                ConvNorm(
                    num_spec_feats + dim_emb if i == 0 else 512,
                    512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(512),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # "Both the forward and backward cell dimensions are 32, so their (LSTMs) combined dimension is 64."
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

        # c_org is speaker embedding

    def forward(self, x, c_org):
        if SVC_model_name == "defaultName":
            pdb.set_trace()
        x = x.squeeze(1).transpose(2, 1)
        # broadcasts c_org to a compatible shape to merge with x
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        saved_enc_outs = [x]  ###
        for conv in self.convolutions:
            x = F.relu(conv(x))
            saved_enc_outs.append(x)  ###
        if SVC_model_name == "defaultName":
            pdb.set_trace()
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        # lstms output 64 dim
        outputs, _ = self.lstm(x)
        saved_enc_outs.append(outputs.transpose(2, 1))  ###
        # backward is the first half of dimensions, forward is the second half
        # pdb.set_trace()
        out_forward = outputs[:, :, : self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck :]

        # pdb.set_trace()
        codes = []
        # for each timestep, skipping self.freq frames
        for i in range(0, outputs.size(1), self.freq):
            # remeber that i is self.freq, not increments of 1)
            codes.append(
                torch.cat(
                    (out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]),
                    dim=-1,
                )
            )
        if SVC_model_name == "defaultName":
            pdb.set_trace()
        # saved_enc_outs.append(codes_cat) ###
        # if self.freq is 32, then codes is a list of 4 tensors of size 64
        return codes, saved_enc_outs


class Decoder(nn.Module):
    """Decoder module:"""

    def __init__(self, dim_neck, dim_emb, dim_pre, num_spec_feats, dim_pitch=0):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(
            dim_neck * 2 + dim_emb + dim_pitch, dim_pre, 1, batch_first=True
        )

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    dim_pre,
                    dim_pre,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(dim_pre),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.linear_projection = LinearNorm(1024, num_spec_feats)

    def forward(self, x):
        # self.lstm1.flatten_parameters()
        if SVC_model_name == "defaultName":
            pdb.set_trace()
        saved_dec_outs = [x.transpose(1, 2)]  ###
        x, _ = self.lstm1(x)
        saved_dec_outs.append(x.transpose(1, 2))  ###
        x = x.transpose(1, 2)
        for conv in self.convolutions:
            x = F.relu(conv(x))
            saved_dec_outs.append(x)  ###
        x = x.transpose(1, 2)
        outputs, _ = self.lstm2(x)
        saved_dec_outs.append(outputs.transpose(1, 2))  ###
        decoder_output = self.linear_projection(outputs)
        saved_dec_outs.append(decoder_output.transpose(1, 2))  ###
        return decoder_output, saved_dec_outs


# Still part of Decoder as indicated in paper Fig. 3 (c) - last two blocks
class Postnet(nn.Module):
    """Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, num_spec_feats):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    num_spec_feats,
                    512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(512),
            )
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        512,
                        512,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(512),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    512,
                    num_spec_feats,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(num_spec_feats),
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, dim_neck, dim_emb, dim_pre, freq, num_spec_feats, dim_pitch=0):
        super(Generator, self).__init__()

        self.encoder = Encoder(dim_neck, dim_emb, freq, num_spec_feats)
        self.decoder = Decoder(
            dim_neck, dim_emb, dim_pre, num_spec_feats, dim_pitch
        )  # dim_pitch not getting value from above dim_pitch
        self.postnet = Postnet(num_spec_feats)

    def forward(self, x, c_org, c_trg, pitch_cont=None):
        if SVC_model_name == "defaultName":
            pdb.set_trace()
        # codes is a LIST of tensors
        codes, saved_enc_outs = self.encoder(x, c_org)
        # if no c_trg given, then just return the formatted encoder codes
        if c_trg is None:
            # concatenates the by stacking over the last (in 2D this would be vertical) dimensio by stacking over the last (in 2D this would be vertical) dimension. For lists it means the same
            return torch.cat(codes, dim=-1)
        # list of reformatted codes
        tmp = []
        for code in codes:
            # reformatting tmp from list to tensor, and resample it at the specified freq
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1) / len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)
        # concat reformated encoder output with target speaker embedding
        encoder_outputs = torch.cat(
            (code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1
        )
        if SVC_model_name == "defaultName":
            pdb.set_trace()
        if (
            pitch_cont != None
        ):  # if pitchCond it activate, concatenate pitch contour with encoder_outputs
            try:
                encoder_outputs = torch.cat((encoder_outputs, pitch_cont), dim=-1)
            except Exception as e:
                pdb.set_trace()

        mel_outputs, saved_dec_outs = self.decoder(encoder_outputs)
        # then put mel_ouputs through remaining postnet section of NN
        # the postnet process produces the RESIDUAL information that gets added to the mel output
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        # pdb.set_trace()
        # add together, as done in Fig. 3 (c) ensuring the mel_out_psnt is same shape (2,128,80). new mel_out_psnt will be the same
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        # insert channel dimension into tensors to become (2,1,128,80)
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return (
            mel_outputs,
            mel_outputs_postnet,
            torch.cat(codes, dim=-1),
            saved_enc_outs,
            saved_dec_outs,
        )


class Aux_Voice_Classifier(nn.Module):
    def __init__(self, input_dim, layer_out_sizes, class_num):
        super(Aux_Voice_Classifier, self).__init__()

        self.fc_layers = nn.ModuleList()
        for layer_out_size in layer_out_sizes:
            fc_layer = nn.Sequential(
                nn.Linear(input_dim, layer_out_size),
                nn.BatchNorm1d(layer_out_size),
                nn.ReLU(),
            )
            self.fc_layers.append(fc_layer)
            input_dim = layer_out_size

        # self.fc1 = nn.Sequential(nn.Linear(input_dim, input_dim//2),
        #                          nn.BatchNorm1d(input_dim//2),
        #                          nn.ReLU()
        #                          )
        self.classlayer = nn.Linear(input_dim, class_num)

    def forward(self, x):
        x.flatten()
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        prediction = torch.sigmoid(self.classlayer(x))
        return prediction


class Disentangle_Eval(nn.Module):
    """Generator network."""

    def __init__(self, dim_neck, dim_emb, freq, num_spec_feats, class_num):
        super(Disentangle_Eval, self).__init__()

        self.encoder = Encoder(dim_neck, dim_emb, freq, num_spec_feats)
        self.classer = Aux_Voice_Classifier((dim_neck * dim_emb), class_num)

    def forward(self, x, c_org, c_trg, pitch_cont=None):
        if SVC_model_name == "defaultName":
            pdb.set_trace()
        # codes is a LIST of tensors
        codes, saved_enc_outs = self.encoder(x, c_org)
        prediction = self.classer(codes)
        return prediction
