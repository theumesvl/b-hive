"""

ParticleTransformer2 models for ttcc analysis
Uses improved architecture with SwiGLU, RMSNorm, and mixed precision
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from utils.models.base_model import Classifier_base
from utils.models.particletransformer2 import (
    InputProcess,
    InputConv,
    PairEmbed,
    HF_TransformerEncoderLayer,
    HF_TransformerEncoder,
    CLS_TransformerEncoderLayer,
    RMSNorm,
    build_E_p,
    trunc_normal_,
)


def build_p4_mass(tensor, is_cpf=False):
    """Build 4-vector from pt, eta, phi, mass instead of energy"""
    out = torch.zeros(tensor.shape[0], tensor.shape[1], 4, device=tensor.device)
    out[:, :, 0] = tensor[:, :, 0] * torch.cos(tensor[:, :, 2])  # Get px
    out[:, :, 1] = tensor[:, :, 0] * torch.sin(tensor[:, :, 2])  # Get py
    out[:, :, 2] = tensor[:, :, 0] * torch.sinh(tensor[:, :, 1])  # Get pz
    p_squared = out[:, :, 0]**2 + out[:, :, 1]**2 + out[:, :, 2]**2
    out[:, :, 3] = torch.sqrt(p_squared + tensor[:, :, 3]**2)  # Get E from mass
    if is_cpf:
        out[:, :, 4:] = tensor[:, :, 4:]
    return out


class InputProcess_ttcc(nn.Module):
    """
    InputProcess for ttcc analysis (cpf + vtx only, no npf)
    Based on ParticleTransformer2's InputProcess but without npf
    """
    def __init__(self, cpf_dim, vtx_dim, embed_dim, **kwargs):
        super(InputProcess_ttcc, self).__init__(**kwargs)
        
        self.cpf_conv1 = InputConv(cpf_dim, embed_dim)
        self.cpf_conv3 = InputConv(embed_dim * 1, embed_dim)

        self.vtx_conv1 = InputConv(vtx_dim, embed_dim)
        self.vtx_conv3 = InputConv(embed_dim * 1, embed_dim)

    def forward(self, cpf, vtx):
        cpf = self.cpf_conv1(cpf, cpf, skip=False)
        cpf = self.cpf_conv3(cpf, cpf, skip=True)

        vtx = self.vtx_conv1(vtx, vtx, skip=False)
        vtx = self.vtx_conv3(vtx, vtx, skip=True)

        out = torch.cat((cpf, vtx), dim=1)

        return out


class ParticleTransformer2_ttcc(Classifier_base):
    """
    ParticleTransformer2 adapted for ttcc event-level classification
    5-class: ttbb, ttb, ttcc, ttc, ttother
    Uses SwiGLU, RMSNorm, and mixed precision for better performance
    """

    classes = {
        "ttbb": ["is_ttbb"],
        "ttb": ["is_ttb"], 
        "ttcc": ["is_ttcc"],
        "ttc": ["is_ttc"],
        "ttother": ["is_ttother"],
    }
    
#    global_features = [
#        "n_jet",
#        "n_btagM",
#        "n_btagT",
#        "n_ctagVL",
#        "n_ctagL",
#        "n_ctagM",
#        "n_ctagT",
#    ]
#        
#    cpf_candidates = [
#        "Jet_tagBin",
#        "Jet_pt",
#        "Jet_eta",
#        "Jet_phi",
#        "Jet_mass",
#    ]
    integer_features = {
        "global_features": ["n_jet", "n_btagVL", "n_btagL", "n_btagM", "n_btagT", "n_ctagVVL", "n_ctagVL", "n_ctagL", "n_ctagM", "n_ctagT", "is_CR1_4j", "is_CR2_4j", "is_CR3_4j", "is_SRl_4j", "is_SRt_4j"],
        "cpf_candidates": ["Jet_tagBin"],
        "npf_candidates": [],
        "vtx_features": [],
    }
        
    def __init__(
        self,
        config,
        num_classes=5,
        num_enc=3,
        num_head=4,
        embed_dim=32,
        for_inference=False,
        build_4v=True,
        swiglu=True,
        use_mass=True,  # Use mass-based 4-vector
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.build_4v = build_4v
        self.use_mass = use_mass
        self.num_enc_layers = num_enc
        self.num_head = num_head
        self.dtype = torch.float16

        global_dim, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()
        
        #cpf_dim = cpf_dim - 4
        #vtx_dim = 2  # Leptons have 2 additional training features
        
        # Use InputProcess_ttcc (cpf + vtx only, no npf)
        self.InputProcess = InputProcess_ttcc(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim+global_dim, num_classes)

        self.pair_embed = PairEmbed(4, [3*num_head,]*2 + [num_head], for_onnx=for_inference)
        self.cls_norm = RMSNorm(embed_dim+global_dim)

        EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        self.Encoder = HF_TransformerEncoder(EncoderLayer, num_layers=num_enc)

        self.CLS_EncoderLayer1 = CLS_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        if self.num_enc_layers > 3:
            self.CLS_EncoderLayer2 = CLS_TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, inpt):
        global_features, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        
        cpf = cpf_features[:, :, :]
        cpf_4v = cpf_features[:, :, -4:]
        
        vtx = vtx_features[:, :, :]
        vtx_4v = vtx_features

        padding_mask = torch.cat((cpf_4v[:, :, :1], vtx_4v[:, :, :1]), dim=1)
        padding_mask = torch.eq(padding_mask[:, :, 0], 0.0)
        pad = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(cpf.dtype) * torch.finfo(self.dtype).min

        if self.build_4v:
            if self.use_mass:
                cpf_4v = build_p4_mass(cpf_4v)
                vtx_4v = build_p4_mass(vtx_4v)
            else:
                cpf_4v = build_E_p(cpf_4v)
                vtx_4v = build_E_p(vtx_4v)

        enc = self.InputProcess(cpf, vtx)  # No npf

        lorentz_vectors = torch.cat((cpf_4v, vtx_4v), dim=1)
        v = lorentz_vectors.transpose(1, 2)
        attn_mask = self.pair_embed(v)
        attn_mask = attn_mask + pad.unsqueeze(dim=1)

        enc = self.Encoder(enc, attn_mask.reshape(-1, v.size(-1), v.size(-1)), padding_mask)

        cls_tokens = self.cls_token.expand(enc.size(0), 1, -1)
        cls_tokens = self.CLS_EncoderLayer1(cls_tokens, enc, padding_mask)
        if self.num_enc_layers > 3:
            cls_tokens = self.CLS_EncoderLayer2(cls_tokens, enc, padding_mask)
        
        x = torch.squeeze(cls_tokens, dim=1)
        x = torch.cat((x,global_features), dim=1)
        output = self.Linear(self.cls_norm(x))
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output


    def calculate_roc_list(self, predictions, truth):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        ttbb_index = list(self.classes).index('ttbb')
        ttb_index = list(self.classes).index('ttb')
        ttcc_index = list(self.classes).index('ttcc')
        ttc_index = list(self.classes).index('ttc')
        ttother_index = list(self.classes).index('ttother')

        ttbb_events = (truth == ttbb_index)
        ttb_events = (truth == ttb_index)
        ttcc_events = (truth == ttcc_index)
        ttc_events = (truth == ttc_index)
        ttother_events = (truth == ttother_index)
        summed_events = ttbb_events + ttb_events + ttcc_events + ttc_events + ttother_events

        ttbb_pred = predictions[:, ttbb_index]
        ttb_pred = predictions[:, ttb_index]
        ttcc_pred = predictions[:, ttcc_index]
        ttc_pred = predictions[:, ttc_index]
        ttother_pred = predictions[:, ttother_index]

        # Key discriminators
        ttcc_vs_ttbb = np.where((ttcc_pred + ttbb_pred) > 0, ttcc_pred / (ttcc_pred + ttbb_pred), -1)
        ttbb_vs_ttcc = np.where((ttcc_pred + ttbb_pred) > 0, ttbb_pred / (ttcc_pred + ttbb_pred), -1)

        ttbb_vs_ttb = np.where((ttbb_pred + ttb_pred) > 0, ttbb_pred / (ttbb_pred + ttb_pred), -1)
        ttcc_vs_ttc = np.where((ttcc_pred + ttc_pred) > 0, ttcc_pred / (ttcc_pred + ttc_pred), -1)

        ttbb_vs_all = np.where(predictions.sum(axis=1) > 0, ttbb_pred / predictions.sum(axis=1), -1)
        ttcc_vs_all = np.where(predictions.sum(axis=1) > 0, ttcc_pred / predictions.sum(axis=1), -1)

        ttbb_vs_ttother = np.where((ttbb_pred + ttother_pred) > 0, ttbb_pred / (ttbb_pred + ttother_pred), -1)
        ttcc_vs_ttother = np.where((ttcc_pred + ttother_pred) > 0, ttcc_pred / (ttcc_pred + ttother_pred), -1)

        ttbb_vs_ttc = np.where((ttbb_pred + ttc_pred) > 0, ttbb_pred / (ttbb_pred + ttc_pred), -1)
        ttcc_vs_ttb = np.where((ttcc_pred + ttb_pred) > 0, ttcc_pred / (ttcc_pred + ttb_pred), -1)

        ttcc_vs_ttbb_veto = (truth != ttb_index) & (truth != ttc_index) & (truth != ttother_index) & (summed_events != 0)
        ttbb_vs_ttb_veto = (truth != ttcc_index) & (truth != ttc_index) & (truth != ttother_index) & (summed_events != 0)
        ttcc_vs_ttc_veto = (truth != ttbb_index) & (truth != ttb_index) & (truth != ttother_index) & (summed_events != 0)
        ttbb_vs_ttother_veto = (truth != ttcc_index) & (truth != ttb_index) & (truth != ttc_index) & (summed_events != 0)
        ttcc_vs_ttother_veto = (truth != ttbb_index) & (truth != ttb_index) & (truth != ttc_index) & (summed_events != 0)
        ttbb_vs_ttc_veto = (truth != ttcc_index) & (truth != ttb_index) & (truth != ttother_index) & (summed_events != 0)
        ttcc_vs_ttb_veto = (truth != ttbb_index) & (truth != ttc_index) & (truth != ttother_index) & (summed_events != 0)
        #no_veto = np.ones(b_veto.shape, dtype=bool)
        no_veto = np.ones(truth.shape, dtype=bool)


        labels = ["ttcc_vs_ttbb", "ttbb_vs_ttcc", "ttbb_vs_ttb", "ttcc_vs_ttc", "ttbb_vs_all", "ttcc_vs_all", "ttbb_vs_ttother", "ttcc_vs_ttother", "ttbb_vs_ttc", "ttcc_vs_ttb"]
        discs = [ttcc_vs_ttbb, ttbb_vs_ttcc, ttbb_vs_ttb, ttcc_vs_ttc, ttbb_vs_all, ttcc_vs_all, ttbb_vs_ttother, ttcc_vs_ttother, ttbb_vs_ttc, ttcc_vs_ttb]
        truths = [ttcc_events, ttbb_events, ttbb_events, ttcc_events, ttbb_events, ttcc_events, ttbb_events, ttcc_events, ttbb_events, ttcc_events]
        vetos = [ttcc_vs_ttbb_veto, ttcc_vs_ttbb_veto, ttbb_vs_ttb_veto, ttcc_vs_ttc_veto, no_veto, no_veto, ttbb_vs_ttother_veto, ttcc_vs_ttother_veto, ttbb_vs_ttc_veto, ttcc_vs_ttb_veto]
        xlabels = [
            "ttcc-identification",
            "ttbb-identification",
            "ttbb-identification",
            "ttcc-identification",
            "ttbb-identification",
            "ttcc-identification",
            "ttbb-identification",
            "ttcc-identification",
            "ttbb-identification",
            "ttcc-identification",
        ]

        ylabels = [
            "ttbb-rejection",
            "ttcc-rejection",
            "ttb-rejection",
            "ttc-rejection",
            "all-rejection",
            "all-rejection",
            "ttother_rejection",
            "ttother_rejection",
            "ttc-rejection",
            "ttb-rejection",
        ]

        return (discs, truths, vetos, labels, xlabels, ylabels)


class ParticleTransformer2_ttcc_binary(Classifier_base):
    """
    Binary classification for ttcc analysis using ParticleTransformer2
    Supports different class grouping strategies
    """

    classes = {
        "ttb": ["is_ttbb", "is_ttb"],
        "ttother": ["is_ttcc", "is_ttc", "is_ttother"],
    }
    
    global_features = [
        "nJet",
        "n_btagM",
        "n_btagT",
        "n_ctagVL",
        "n_ctagL",
        "n_ctagM",
        "n_ctagT",
    ]
        
    cpf_candidates = [
        "Jet_tagBin",
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_mass",
    ]
    integer_features = {
        "global_features": ["nJet", "n_btagT", "n_btagM", "n_ctagVL", "n_ctagL", "n_ctagM", "n_ctagT"],
        "cpf_candidates": ["Jet_tagBin"],
        "npf_candidates": [],
        "vtx_features": [],
    }
        
    def __init__(
        self,
        config,
        num_classes=2,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        swiglu=True,
        use_mass=True,
        class_grouping='ttb_vs_ttother',
        **kwargs
    ):
        self.class_grouping = class_grouping
        if class_grouping == 'ttb_vs_ttother':
            self.classes = {
                "ttb": ["is_ttbb", "is_ttb"],
                "ttother": ["is_ttcc", "is_ttc", "is_ttother"],
            }
        elif class_grouping == 'ttc_vs_ttother':
            self.classes = {
                "ttc": ["is_ttcc", "is_ttc"],
                "ttother": ["is_ttbb", "is_ttb", "is_ttother"],
            }
        else:
            raise ValueError(f"Unknown class_grouping: {class_grouping}")
        
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.build_4v = build_4v
        self.use_mass = use_mass
        self.num_enc_layers = num_enc
        self.num_head = num_head
        self.dtype = torch.float16

        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()
        cpf_dim = cpf_dim - 4
        vtx_dim = 2
        
        self.InputProcess = InputProcess_ttcc(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)

        self.pair_embed = PairEmbed(4, [3*num_head,]*2 + [num_head], for_onnx=for_inference)
        self.cls_norm = RMSNorm(embed_dim)

        EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        self.Encoder = HF_TransformerEncoder(EncoderLayer, num_layers=num_enc)

        self.CLS_EncoderLayer1 = CLS_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        if self.num_enc_layers > 3:
            self.CLS_EncoderLayer2 = CLS_TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, inpt):
        _, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        
        cpf = cpf_features[:, :, :-4]
        cpf_4v = cpf_features[:, :, -4:]
        
        vtx = vtx_features[:, :, :-2]
        vtx_4v = vtx_features

        padding_mask = torch.cat((cpf_4v[:, :, :1], vtx_4v[:, :, :1]), dim=1)
        padding_mask = torch.eq(padding_mask[:, :, 0], 0.0)
        pad = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(cpf.dtype) * torch.finfo(self.dtype).min

        if self.build_4v:
            if self.use_mass:
                cpf_4v = build_p4_mass(cpf_4v)
                vtx_4v = build_p4_mass(vtx_4v)
            else:
                cpf_4v = build_E_p(cpf_4v)
                vtx_4v = build_E_p(vtx_4v)

        enc = self.InputProcess(cpf, npf, vtx)

        lorentz_vectors = torch.cat((cpf_4v, vtx_4v), dim=1)
        v = lorentz_vectors.transpose(1, 2)
        attn_mask = self.pair_embed(v)
        attn_mask = attn_mask + pad.unsqueeze(dim=1)

        enc = self.Encoder(enc, attn_mask.reshape(-1, v.size(-1), v.size(-1)), padding_mask)

        cls_tokens = self.cls_token.expand(enc.size(0), 1, -1)
        cls_tokens = self.CLS_EncoderLayer1(cls_tokens, enc, padding_mask)
        if self.num_enc_layers > 3:
            cls_tokens = self.CLS_EncoderLayer2(cls_tokens, enc, padding_mask)
        
        x = torch.squeeze(cls_tokens, dim=1)
        output = self.Linear(self.cls_norm(x))
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output

    def calculate_roc_list(self, predictions, truth):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        class_names = list(self.classes.keys())
        class1_name = class_names[0]
        class2_name = class_names[1]
        
        class1_index = list(self.classes).index(class1_name)
        class2_index = list(self.classes).index(class2_name)
        
        class1_events = (truth == class1_index)
        class2_events = (truth == class2_index)

        class1_pred = predictions[:, class1_index]
        class2_pred = predictions[:, class2_index]

        class1_vs_class2 = np.where((class1_pred + class2_pred) > 0, 
                                    class1_pred / (class1_pred + class2_pred), -1)

        no_veto = np.ones(truth.shape, dtype=bool)

        labels = [f"{class1_name}_vs_{class2_name}", f"{class2_name}_vs_{class1_name}"]
        discs = [class1_vs_class2, 1 - class1_vs_class2]
        truths = [class1_events, class2_events]
        vetos = [no_veto, no_veto]
        xlabels = [f"{class1_name}-identification", f"{class2_name}-identification"]
        ylabels = [f"{class2_name}-rejection", f"{class1_name}-rejection"]

        return (discs, truths, vetos, labels, xlabels, ylabels)


class ParticleTransformer2_ttb_binary(ParticleTransformer2_ttcc_binary):
    """Binary classification: ttb (ttbb+ttb) vs ttother (ttcc+ttc+ttother)"""
    def __init__(self, config, **kwargs):
        super().__init__(config, class_grouping='ttb_vs_ttother', **kwargs)


class ParticleTransformer2_ttc_binary(ParticleTransformer2_ttcc_binary):
    """Binary classification: ttc (ttcc+ttc) vs ttother (ttbb+ttb+ttother)"""
    def __init__(self, config, **kwargs):
        super().__init__(config, class_grouping='ttc_vs_ttother', **kwargs)


class ParticleTransformer2_ttcc_3class(Classifier_base):
    """
    3-class classification: ttb (ttbb+ttb), ttc (ttcc+ttc), ttother
    Uses ParticleTransformer2 architecture
    """

    classes = {
        "ttb": ["is_ttbb", "is_ttb"],
        "ttc": ["is_ttcc", "is_ttc"],
        "ttother": ["is_ttother"],
    }

    global_features = [
        "nJet",
        "n_btagM",
        "n_btagT",
        "n_ctagVL",
        "n_ctagL",
        "n_ctagM",
        "n_ctagT",
    ]
        
    cpf_candidates = [
        "Jet_tagBin",
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_mass",
    ]
    
    integer_features = {
        "global_features": ["nJet", "n_btagT", "n_btagM", "n_ctagVL", "n_ctagL", "n_ctagM", "n_ctagT"],
        "cpf_candidates": ["Jet_tagBin"],
        "npf_candidates": [],
        "vtx_features": [],
    }
        
    def __init__(
        self,
        config,
        num_classes=3,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        swiglu=True,
        use_mass=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.build_4v = build_4v
        self.use_mass = use_mass
        self.num_enc_layers = num_enc
        self.num_head = num_head
        self.dtype = torch.float16

        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()
        cpf_dim = cpf_dim - 4
        vtx_dim = 2
        
        self.InputProcess = InputProcess_ttcc(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)

        self.pair_embed = PairEmbed(4, [3*num_head,]*2 + [num_head], for_onnx=for_inference)
        self.cls_norm = RMSNorm(embed_dim)

        EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        self.Encoder = HF_TransformerEncoder(EncoderLayer, num_layers=num_enc)

        self.CLS_EncoderLayer1 = CLS_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        if self.num_enc_layers > 3:
            self.CLS_EncoderLayer2 = CLS_TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, inpt):
        _, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        
        cpf = cpf_features[:, :, :-4]
        cpf_4v = cpf_features[:, :, -4:]
        
        vtx = vtx_features[:, :, :-2]
        vtx_4v = vtx_features

        padding_mask = torch.cat((cpf_4v[:, :, :1], vtx_4v[:, :, :1]), dim=1)
        padding_mask = torch.eq(padding_mask[:, :, 0], 0.0)
        pad = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(cpf.dtype) * torch.finfo(self.dtype).min

        if self.build_4v:
            if self.use_mass:
                cpf_4v = build_p4_mass(cpf_4v)
                vtx_4v = build_p4_mass(vtx_4v)
            else:
                cpf_4v = build_E_p(cpf_4v)
                vtx_4v = build_E_p(vtx_4v)

        enc = self.InputProcess(cpf, npf, vtx)

        lorentz_vectors = torch.cat((cpf_4v, vtx_4v), dim=1)
        v = lorentz_vectors.transpose(1, 2)
        attn_mask = self.pair_embed(v)
        attn_mask = attn_mask + pad.unsqueeze(dim=1)

        enc = self.Encoder(enc, attn_mask.reshape(-1, v.size(-1), v.size(-1)), padding_mask)

        cls_tokens = self.cls_token.expand(enc.size(0), 1, -1)
        cls_tokens = self.CLS_EncoderLayer1(cls_tokens, enc, padding_mask)
        if self.num_enc_layers > 3:
            cls_tokens = self.CLS_EncoderLayer2(cls_tokens, enc, padding_mask)
        
        x = torch.squeeze(cls_tokens, dim=1)
        output = self.Linear(self.cls_norm(x))
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output

    def calculate_roc_list(self, predictions, truth):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        class_names = list(self.classes.keys())
        class1_name = class_names[0]
        class2_name = class_names[1]
        
        class1_index = list(self.classes).index(class1_name)
        class2_index = list(self.classes).index(class2_name)
        
        class1_events = (truth == class1_index)
        class2_events = (truth == class2_index)

        class1_pred = predictions[:, class1_index]
        class2_pred = predictions[:, class2_index]

        class1_vs_class2 = np.where((class1_pred + class2_pred) > 0, 
                                    class1_pred / (class1_pred + class2_pred), -1)

        no_veto = np.ones(truth.shape, dtype=bool)

        labels = [f"{class1_name}_vs_{class2_name}", f"{class2_name}_vs_{class1_name}"]
        discs = [class1_vs_class2, 1 - class1_vs_class2]
        truths = [class1_events, class2_events]
        vetos = [no_veto, no_veto]
        xlabels = [f"{class1_name}-identification", f"{class2_name}-identification"]
        ylabels = [f"{class2_name}-rejection", f"{class1_name}-rejection"]

        return (discs, truths, vetos, labels, xlabels, ylabels)


class ParticleTransformer2_ttb_binary(ParticleTransformer2_ttcc_binary):
    """Binary classification: ttb (ttbb+ttb) vs ttother (ttcc+ttc+ttother)"""
    def __init__(self, config, **kwargs):
        super().__init__(config, class_grouping='ttb_vs_ttother', **kwargs)


class ParticleTransformer2_ttc_binary(ParticleTransformer2_ttcc_binary):
    """Binary classification: ttc (ttcc+ttc) vs ttother (ttbb+ttb+ttother)"""
    def __init__(self, config, **kwargs):
        super().__init__(config, class_grouping='ttc_vs_ttother', **kwargs)


class ParticleTransformer2_ttcc_3class(Classifier_base):
    """
    3-class classification: ttb (ttbb+ttb), ttc (ttcc+ttc), ttother
    Uses ParticleTransformer2 architecture
    """

    classes = {
        "ttb": ["is_ttbb", "is_ttb"],
        "ttc": ["is_ttcc", "is_ttc"],
        "ttother": ["is_ttother"],
    }
    
    integer_features = {
        "global_features": ["nJet", "n_btagT", "n_btagM", "n_ctagL", "n_ctagM", "n_ctagT"],
        "cpf_candidates": [],
        "npf_candidates": [],
        "vtx_features": [],
    }
        
    def __init__(
        self,
        config,
        num_classes=3,
        num_enc=3,
        num_head=8,
        embed_dim=128,
        for_inference=False,
        build_4v=True,
        swiglu=True,
        use_mass=True,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        
        self.for_inference = for_inference
        self.build_4v = build_4v
        self.use_mass = use_mass
        self.num_enc_layers = num_enc
        self.num_head = num_head
        self.dtype = torch.float16

        _, cpf_dim, _, vtx_dim, _ = self.all_model_features_length()
        cpf_dim = cpf_dim - 4
        vtx_dim = 2
        
        self.InputProcess = InputProcess_ttcc(cpf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)

        self.pair_embed = PairEmbed(4, [3*num_head,]*2 + [num_head], for_onnx=for_inference)
        self.cls_norm = RMSNorm(embed_dim)

        EncoderLayer = HF_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        self.Encoder = HF_TransformerEncoder(EncoderLayer, num_layers=num_enc)

        self.CLS_EncoderLayer1 = CLS_TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
        )
        if self.num_enc_layers > 3:
            self.CLS_EncoderLayer2 = CLS_TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_head, dropout=0.1, swiglu=swiglu,
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, inpt):
        _, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        
        cpf = cpf_features[:, :, :-4]
        cpf_4v = cpf_features[:, :, -4:]
        
        vtx = vtx_features[:, :, :-2]
        vtx_4v = vtx_features

        padding_mask = torch.cat((cpf_4v[:, :, :1], vtx_4v[:, :, :1]), dim=1)
        padding_mask = torch.eq(padding_mask[:, :, 0], 0.0)
        pad = (padding_mask.unsqueeze(1) + padding_mask.unsqueeze(2)).to(cpf.dtype) * torch.finfo(self.dtype).min

        if self.build_4v:
            if self.use_mass:
                cpf_4v = build_p4_mass(cpf_4v)
                vtx_4v = build_p4_mass(vtx_4v)
            else:
                cpf_4v = build_E_p(cpf_4v)
                vtx_4v = build_E_p(vtx_4v)

        enc = self.InputProcess(cpf, npf, vtx)

        lorentz_vectors = torch.cat((cpf_4v, vtx_4v), dim=1)
        v = lorentz_vectors.transpose(1, 2)
        attn_mask = self.pair_embed(v)
        attn_mask = attn_mask + pad.unsqueeze(dim=1)

        enc = self.Encoder(enc, attn_mask.reshape(-1, v.size(-1), v.size(-1)), padding_mask)

        cls_tokens = self.cls_token.expand(enc.size(0), 1, -1)
        cls_tokens = self.CLS_EncoderLayer1(cls_tokens, enc, padding_mask)
        if self.num_enc_layers > 3:
            cls_tokens = self.CLS_EncoderLayer2(cls_tokens, enc, padding_mask)
        
        x = torch.squeeze(cls_tokens, dim=1)
        output = self.Linear(self.cls_norm(x))
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output

    def calculate_roc_list(self, predictions, truth):
        if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
            predictions = softmax(predictions, axis=-1)

        ttb_index = list(self.classes).index('ttb')
        ttc_index = list(self.classes).index('ttc')
        ttother_index = list(self.classes).index('ttother')
        
        ttb_events = (truth == ttb_index)
        ttc_events = (truth == ttc_index)
        ttother_events = (truth == ttother_index)

        ttb_pred = predictions[:, ttb_index]
        ttc_pred = predictions[:, ttc_index]
        ttother_pred = predictions[:, ttother_index]

        # Key discriminators
        ttb_vs_ttc = np.where((ttb_pred + ttc_pred) > 0, ttb_pred / (ttb_pred + ttc_pred), -1)
        ttc_vs_ttb = np.where((ttb_pred + ttc_pred) > 0, ttc_pred / (ttb_pred + ttc_pred), -1)
        ttb_vs_ttother = np.where((ttb_pred + ttother_pred) > 0, ttb_pred / (ttb_pred + ttother_pred), -1)
        ttc_vs_ttother = np.where((ttc_pred + ttother_pred) > 0, ttc_pred / (ttc_pred + ttother_pred), -1)
        
        ttb_vs_all = np.where(predictions.sum(axis=1) > 0, ttb_pred / predictions.sum(axis=1), -1)
        ttc_vs_all = np.where(predictions.sum(axis=1) > 0, ttc_pred / predictions.sum(axis=1), -1)

        no_veto = np.ones(truth.shape, dtype=bool)

        labels = ["ttb_vs_ttc", "ttc_vs_ttb", "ttb_vs_ttother", "ttc_vs_ttother", "ttb_vs_all", "ttc_vs_all"]
        discs = [ttb_vs_ttc, ttc_vs_ttb, ttb_vs_ttother, ttc_vs_ttother, ttb_vs_all, ttc_vs_all]
        truths = [ttb_events, ttc_events, ttb_events, ttc_events, ttb_events, ttc_events]
        vetos = [no_veto, no_veto, no_veto, no_veto, no_veto, no_veto]
        xlabels = [
            "ttb-identification",
            "ttc-identification",
            "ttb-identification",
            "ttc-identification",
            "ttb-identification",
            "ttc-identification",
        ]
        ylabels = [
            "ttc-rejection",
            "ttb-rejection",
            "ttother-rejection",
            "ttother-rejection",
            "all-rejection",
            "all-rejection",
        ]

        return (discs, truths, vetos, labels, xlabels, ylabels)
