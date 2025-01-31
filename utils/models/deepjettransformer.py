import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from functools import partial
import numpy as np
from typing import List
from scipy.special import softmax

from utils.models.base_model import Classifier_base


class InputConv(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(InputConv, self).__init__(**kwargs)
        
        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip: bool = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = sc + x2
        else:
            x = x2
        return x
    
class LinLayer(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)
        
        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.bn2 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = self.bn2(sc + x2)
        else:
            x = self.bn2(x2)
        return x

class LinLayer2(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(LinLayer2, self).__init__(**kwargs)

        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.ln = torch.nn.LayerNorm(out_chn, eps = 0.001)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.dropout(self.ln(self.act(self.lin(x))))
        return x

class InputProcess(nn.Module):

    def __init__(self, cpf_dim, npf_dim, vtx_dim, embed_dim, **kwargs):
        super(InputProcess, self).__init__(**kwargs)
        
        self.cpf_bn0 = torch.nn.BatchNorm1d(cpf_dim, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(cpf_dim,embed_dim)
        self.cpf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.npf_bn0 = torch.nn.BatchNorm1d(npf_dim, eps = 0.001, momentum = 0.1)
        self.npf_conv1 = InputConv(npf_dim,embed_dim)
        self.npf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
        self.vtx_conv3 = InputConv(embed_dim*1,embed_dim)


    def forward(self, cpf, npf, vtx):
                
        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        npf = self.npf_bn0(torch.transpose(npf, 1, 2))
        npf = self.npf_conv1(npf, npf, skip = False)
        npf = self.npf_conv3(npf, npf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,npf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out

class InputProcess_HLT(nn.Module):

    def __init__(self, cpf_dim, vtx_dim, embed_dim, **kwargs):
        super(InputProcess_HLT, self).__init__(**kwargs)

        self.cpf_bn0 = torch.nn.BatchNorm1d(cpf_dim, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(cpf_dim,embed_dim)
        self.cpf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
        self.vtx_conv3 = InputConv(embed_dim*1,embed_dim)

    def forward(self, cpf, vtx):

        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out
        
class DenseClassifier(nn.Module):

    def __init__(self, embed_dim, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)
             
        self.LinLayer1 = LinLayer(embed_dim, embed_dim)

    def forward(self, x):
        
        x = self.LinLayer1(x, x, skip = True)
        
        return x
    
class AttentionPooling(nn.Module):

    def __init__(self, dim = 128, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)

        self.ConvLayer = torch.nn.Conv1d(dim, 1, kernel_size=1)
        self.Softmax = nn.Softmax(dim=-1)
        self.bn = torch.nn.BatchNorm1d(dim, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        
        a = self.ConvLayer(torch.transpose(x, 1, 2))
        a = self.Softmax(a)
        
        y = torch.matmul(a,x)
        y = torch.squeeze(y, dim = 1)
        y = self.dropout(self.bn(self.act(y)))
        
        return y
    
class HF_TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
       >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super(HF_TransformerEncoderLayer, self).__init__()
        #MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*4, d_model)

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model*4)
        self.dropout0 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.GELU()
        super(HF_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, padding_mask):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.norm0(src)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            src2 = self.self_attn(src2, src2, src2)[0] #, key_padding_mask = padding_mask)[0]
        src = src + src2
        src = self.norm1(src)
        
        src2 = self.dropout0(self.linear2(self.norm2(self.activation(self.linear1(src)))))
        src = src + src2
        return src
    
class HF_TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super(HF_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, padding_mask):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src
        padding_mask = padding_mask

        for mod in self.layers:
            output = mod(output, padding_mask)

        return output
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.ReLU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class DeepJetTransformer(Classifier_base):

    cpf_candidates = [
        "Cpfcan_ptrel",
        "Cpfcan_drminsv",
        "Cpfcan_VTX_ass",
        "Cpfcan_quality",
        "Cpfcan_pt",
        "Cpfcan_eta",
        "Cpfcan_puppiw",
        "Cpfcan_chi2",
        "Cpfcan_phi",
        "Cpfcan_e"
    ]
    
    npf_candidates = [
        "Npfcan_ptrel"
    ]
    
    vtx_features = [
        "sv_deltaR",
        "sv_normchi2",
        "sv_dxy",
        "sv_eta",
        "sv_phi",
        "sv_mass",
        "sv_ntracks",
        "sv_chi2",
        "sv_e"
    ]

    def __init__(
        self,
        config,
        num_classes=6,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config
        self.for_inference = for_inference
        self.num_enc_layers = num_enc

        _, cpf_dim, npf_dim, vtx_dim = self.all_model_features_length()  
        
        self.InputProcess = InputProcess(cpf_dim, npf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)
        self.DenseClassifier = DenseClassifier(embed_dim)
        self.Pooling = AttentionPooling()

        self.EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1
        )
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)
    
    def forward(self, inpt):

        global_features, cpf_features, npf_features, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        cpf, npf, vtx = cpf_features, npf_features, vtx_features
        
        padding_mask = torch.cat((cpf[:,:,:1],npf[:,:,:1],vtx[:,:,:1]), dim = 1)
        padding_mask = torch.eq(padding_mask[:,:,0], 0.0)
        
        enc = self.InputProcess(cpf, npf, vtx)
        enc = self.Encoder(enc, padding_mask)
        enc = self.Pooling(enc)
        
        x = self.DenseClassifier(enc)
        output = self.Linear(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output


class DeepJetTransformerHLT(Classifier_base):

    # Define classes used by model and connect it to the config definitions
    classes = {
        "b": ["label_b", "label_gbb", "label_bb", "label_leptonicb", "label_leptonicbc"],
        "c": ["label_c", "label_gcc", "label_cc"],
        "uds": ["label_ud", "label_s"],
        "g": ["label_g"],
        "taup": ["label_taup"],
        "taum": ["label_taum"],
    }
    cpf_candidates = [
        "jet_pfcand_deta",
        "jet_pfcand_dphi",
        "jet_pfcand_pt_log",
        "jet_pfcand_energy_log",
        "jet_pfcand_eta",
        "jet_pfcand_charge",
        "jet_pfcand_frompv",
        "jet_pfcand_nlostinnerhits",
        "jet_pfcand_track_chi2",
        "jet_pfcand_track_qual",
        "jet_pfcand_dz",
        "jet_pfcand_dzsig",
        "jet_pfcand_dxy",
        "jet_pfcand_dxysig",
        "jet_pfcand_etarel",
        "jet_pfcand_pperp_ratio",
        "jet_pfcand_ppara_ratio",
        "jet_pfcand_trackjet_d3d",
        "jet_pfcand_trackjet_d3dsig",
        "jet_pfcand_trackjet_dist",
        "jet_pfcand_trackjet_decayL",
        "jet_pfcand_npixhits",
        "jet_pfcand_nstriphits",
        #"pfcand_mask"
    ]
    vtx_features = [
        "jet_sv_deta",
        "jet_sv_dphi",
        "jet_sv_pt_log",
        "jet_sv_mass",
        "jet_sv_eta",
        "jet_sv_ntrack",
        "jet_sv_chi2",
        "jet_sv_dxy",
        "jet_sv_dxysig",
        "jet_sv_d3d",
        "jet_sv_d3dsig",
        #"sv_mask"
    ]
    # Features that have integer dtype and should not be changed during adv attacks
    integer_features = {
        "global_features": ["jet_elf", "jet_muf", "jet_ncand", "jet_nbhad", "jet_nchad"],
        "cpf_candidates": ["jet_pfcand_calofraction", "jet_pfcand_hcalfraction", "jet_pfcand_frompv", "jet_pfcand_id",
                             "jet_pfcand_charge", "jet_pfcand_track_qual", "jet_pfcand_track_chi2", "jet_pfcand_npixhits", 
                             "jet_pfcand_nstriphits", "jet_pfcand_nlostinnerhits"],
        "npf_candidates": [],
        "vtx_features": ["jet_sv_ntrack"],
    }

    def get_inpt(self, x, truth=None, loss_fn=None, attack=None, device='cpu'):

        glob, cpf, npf, vtx = x.split(self.feature_lengths.tolist(), dim=1)
        
        glob = glob.reshape(glob.shape[0], -1)
        cpf = cpf.reshape(cpf.shape[0], self.input_dims[1][0], -1)
        npf = npf.reshape(npf.shape[0], self.input_dims[2][0], 0)
        vtx = vtx.reshape(vtx.shape[0], self.input_dims[3][0], -1)

        # Subselect features
        glob = self._subselect_features(glob, "global_features")
        cpf = self._subselect_features(cpf, "cpf_candidates")
        npf = self._subselect_features(npf, "npf_candidates")
        vtx = self._subselect_features(vtx, "vtx_features")

        if attack is not None:
            (
                glob,
                cpf,
                npf,
                vtx,
                truth,
            ) = attack(
                [
                    feature.float().to(device)
                    for feature in [
                        glob,
                        cpf,
                        npf,
                        vtx,
                        ]
                    ],
                truth.type(torch.LongTensor).to(device),
                self,
                loss_fn,
                device,
            ) 
        
        return (glob.detach(), cpf.detach(), npf.detach(), vtx.detach()), truth

    def __init__(
        self,
        config,
        num_classes=6,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.config = config
        self.for_inference = for_inference
        self.num_enc_layers = num_enc

        _, cpf_dim, _, vtx_dim = self.all_model_features_length()  
        
        self.InputProcess = InputProcess_HLT(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)
        self.DenseClassifier = DenseClassifier(embed_dim)
        self.Pooling = AttentionPooling()

        self.EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1
        )
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)
    
    def forward(self, inpt):

        global_features, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        cpf, vtx = cpf_features, vtx_features

        padding_mask = torch.cat((cpf[:,:,:1],vtx[:,:,:1]), dim = 1)
        padding_mask = torch.eq(padding_mask[:,:,0], 0.0)
        
        enc = self.InputProcess(cpf, vtx)
        enc = self.Encoder(enc, padding_mask)
        enc = self.Pooling(enc)
        
        x = self.DenseClassifier(enc)
        output = self.Linear(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output
        
    def calculate_roc_list(
        self,
        predictions,
        truth,
    ):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        b_index   = list(self.classes).index('b')
        c_index   = list(self.classes).index('c')
        uds_index = list(self.classes).index('uds')
        g_index   = list(self.classes).index('g')
        taup_index = list(self.classes).index('taup')
        taum_index = list(self.classes).index('taum')
        
        b_jets   = (truth == b_index)
        tau_jets = (truth == taup_index) | (truth == taum_index)
        c_jets   = (truth == c_index)
        uds_jets = (truth == uds_index)
        g_jets   = (truth == g_index)
        l_jets   = uds_jets | g_jets
        summed_jets = b_jets + c_jets + l_jets + tau_jets

        b_pred = predictions[:, b_index]
        taup_pred = predictions[:, taup_index]
        taum_pred = predictions[:, taum_index]
        c_pred = predictions[:, c_index]
        uds_pred = predictions[:, uds_index]
        g_pred = predictions[:, g_index]
        l_pred = predictions[:, [uds_index, g_index]].sum(axis=1)

        bvsl = np.where((b_pred + l_pred) > 0, (b_pred) / (b_pred + l_pred), -1)
        bvsc = np.where((b_pred + c_pred) > 0, (b_pred) / (b_pred + c_pred), -1)
        cvsb = np.where((b_pred + c_pred) > 0, (c_pred) / (b_pred + c_pred), -1)
        cvsl = np.where((l_pred + c_pred) > 0, (c_pred) / (l_pred + c_pred), -1)
        bvsall = np.where(
            (b_pred + l_pred + c_pred + taup_pred + taum_pred) > 0, (b_pred) / (b_pred + l_pred + c_pred + taup_pred + taum_pred), -1
        )
        uds_vs_g = np.where((uds_pred + g_pred) > 0, (uds_pred) / (uds_pred + g_pred), -1)

        tau_veto = (truth != taup_index) & (truth != taum_index) & (summed_jets != 0)
        b_veto = (truth != b_index) & tau_veto & (summed_jets != 0)
        c_veto = (truth != c_index) & tau_veto & (summed_jets != 0)
        bc_veto = b_veto & c_veto
        l_veto = (truth != uds_index) & (truth != g_index) & tau_veto & (summed_jets != 0)
        no_veto = np.ones(b_veto.shape, dtype=bool)

        labels = ["bvsl", "bvsc", "cvsb", "cvsl", "bvsall", "uds_vs_g"]
        discs = [bvsl, bvsc, cvsb, cvsl, bvsall, uds_vs_g]
        vetos = [c_veto, l_veto, l_veto, b_veto, no_veto, bc_veto]
        truths = [b_jets, b_jets, c_jets, c_jets, b_jets, uds_jets]
        xlabels = [
            "b-identification",
            "b-identification",
            "c-identification",
            "c-identification",
            "b-identification",
            "uds-identification",
        ]
        ylabels = ["light mis-id.", "c mis-id", "b mis-id.", "light mis-id.", "mis-id.", "gluons mis-id."]

        return discs, truths, vetos, labels, xlabels, ylabels



class GlobalDeepJetTransformerHLT(DeepJetTransformerHLT):
        
    def __init__(
        self,
        config,
        num_classes=6,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.config = config
        self.for_inference = for_inference
        self.num_enc_layers = num_enc

        global_dim, cpf_dim, _, vtx_dim = self.all_model_features_length()
        
        self.global_bn = torch.nn.BatchNorm1d(global_dim, eps=0.001, momentum=0.6)
        self.InputProcess = InputProcess_HLT(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim + global_dim, num_classes)
        self.DenseClassifier = DenseClassifier(embed_dim + global_dim)
        self.Pooling = AttentionPooling()

        self.EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1
        )
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)
    
    def forward(self, inpt):

        global_features, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        cpf, vtx = cpf_features, vtx_features

        global_features = self.global_bn(global_features)

        padding_mask = torch.cat((cpf[:,:,:1],vtx[:,:,:1]), dim = 1)
        padding_mask = torch.eq(padding_mask[:,:,0], 0.0)
        
        enc = self.InputProcess(cpf, vtx)
        enc = self.Encoder(enc, padding_mask)
        enc = self.Pooling(enc)

        fts = torch.cat((global_features, enc), dim=1)
        
        x = self.DenseClassifier(fts)
        output = self.Linear(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output