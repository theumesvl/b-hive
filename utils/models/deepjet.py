import numpy as np
<<<<<<< HEAD
import torch
import torch.nn as nn
from scipy.special import softmax

from utils.models.base_model import Classifier_base
from utils.models.helpers import DenseClassifier, InputProcess, InputProcess_HLT

torch.multiprocessing.set_sharing_strategy("file_system")

class DeepJet(Classifier_base):
=======
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.special import softmax
import torch
from tqdm import tqdm
from utils.models.abstract_base_models import Classifier
from utils.models.helpers import (
    DenseClassifier,
    InputProcess,
    MoDDenseClassifier,
    MoDInputProcess,
)
from utils.plotting.termplot import terminal_roc
from utils.torch import DeepJetDataset


torch.multiprocessing.set_sharing_strategy("file_system")


class DeepJet(Classifier, torch.nn.Module):
    n_cpf = 25
    n_npf = 25
    n_vtx = 5
    datasetClass = DeepJetDataset
    optimizerClass = torch.optim.Adam
>>>>>>> master

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isUD", "isS"],
        "g": ["isG"],
    }

    integer_features = {
            "global_features": ["n_Cpfcand", "nCpfcan", "n_Npfcand", "nNpfcan", "nsv", "npv",
                                "TagVarCSV_vertexCategory", "TagVarCSV_jetNSelectedTracks", "TagVarCSV_jetNTracksEtaRel"],
            "cpf_candidates": ["Cpfcan_VTX_ass", "Cpfcan_puppiw", "Cpfcan_chi2", "Cpfcan_quality"],
            "npf_candidates": ["Npfcan_isGamma", "Npfcan_HadFrac", "Npfcan_puppiw"],
            "vtx_features": ["sv_ntracks"]
    }

    def __init__(self,
                 config,
                 cpf_conv = [64, 32, 32, 8],
                 npf_conv = [32, 16, 4],
                 vtx_conv = [64, 32, 32, 8],
                 n_layers_lstm = 1,
                 lstm_dim = [150, 50, 50],
                 dense_clas_dim = [200, 100, 100, 100, 100, 100, 100, 100, 100],
                 **kwargs
        ):
        
        super().__init__(**kwargs)

        self.config = config

        global_dim, cpf_dim, npf_dim, vtx_dim = self.all_model_features_length()  
        cpf_conv = [cpf_dim] + cpf_conv
        npf_conv = [npf_dim] + npf_conv
        vtx_conv = [vtx_dim] + vtx_conv
        
        self.InputProcess = InputProcess(cpf_conv, npf_conv, vtx_conv)

        dense_clas_dim_full = [sum(lstm_dim) + global_dim] + dense_clas_dim
        self.DenseClassifier = DenseClassifier(dense_clas_dim_full)

<<<<<<< HEAD
        self.global_bn = torch.nn.BatchNorm1d(global_dim, eps=0.001, momentum=0.6)
=======
    def __init__(self, feature_edges=[15, 415, 565, 625], **kwargs):
        super(DeepJet, self).__init__(**kwargs)

        self.feature_edges = np.array(feature_edges)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.InputProcess = InputProcess()
        self.DenseClassifier = DenseClassifier()

        self.global_bn = torch.nn.BatchNorm1d(15, eps=0.001, momentum=0.6)
>>>>>>> master
        self.cpf_lstm = torch.nn.LSTM(
            input_size=cpf_conv[-1], hidden_size=lstm_dim[0], num_layers=n_layers_lstm, batch_first=True
        )
        self.npf_lstm = torch.nn.LSTM(
            input_size=npf_conv[-1], hidden_size=lstm_dim[1],  num_layers=n_layers_lstm, batch_first=True
        )
        self.vtx_lstm = torch.nn.LSTM(
            input_size=vtx_conv[-1], hidden_size=lstm_dim[2],  num_layers=n_layers_lstm, batch_first=True
        )

        self.cpf_bn = torch.nn.BatchNorm1d(lstm_dim[0], eps=0.001, momentum=0.6)
        self.npf_bn = torch.nn.BatchNorm1d(lstm_dim[1], eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(lstm_dim[2], eps=0.001, momentum=0.6)

        self.cpf_dropout = torch.nn.Dropout(0.1)
        self.npf_dropout = torch.nn.Dropout(0.1)
        self.vtx_dropout = torch.nn.Dropout(0.1)

<<<<<<< HEAD
        self.Linear = nn.Linear(100, len(self.classes))
        
    def forward(self, inpt):
         
        global_features, cpf_features, npf_features, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
        
=======
        self.Linear = torch.nn.Linear(100, len(self.classes))

        self.glob_integers = torch.tensor([2, 3, 4, 5, 8, 13, 14])
        self.cpf_integers = torch.tensor([12, 13, 14, 15])
        self.npf_integers = torch.tensor([2])
        self.vtx_integers = torch.tensor([3])
        self.integers = [
            self.glob_integers,
            self.cpf_integers,
            self.npf_integers,
            self.vtx_integers,
        ]
        self.glob_defaults = torch.tensor([0])
        self.cpf_defaults = torch.tensor([0])
        self.npf_defaults = torch.tensor([0])
        self.vtx_defaults = torch.tensor([0])
        self.defaults = [
            self.glob_defaults,
            self.cpf_defaults,
            self.npf_defaults,
            self.vtx_defaults,
        ]

    def forward(self, global_features, cpf_features, npf_features, vtx_features):
>>>>>>> master
        global_features = self.global_bn(global_features)
        
        cpf, npf, vtx = self.InputProcess(cpf_features, npf_features, vtx_features)
        
        cpf = self.cpf_lstm(torch.flip(cpf, dims=[1]))[0][:, -1]
        cpf = self.cpf_dropout(self.cpf_bn(cpf))
            
        npf = self.npf_lstm(torch.flip(npf, dims=[1]))[0][:, -1]
        npf = self.npf_dropout(self.npf_bn(npf))

        vtx = self.vtx_lstm(torch.flip(vtx, dims=[1]))[0][:, -1]
        vtx = self.vtx_dropout(self.vtx_bn(vtx))

        fts = torch.cat((global_features, cpf, npf, vtx), dim=1)
        fts = self.DenseClassifier(fts)

        output = self.Linear(fts)

        return output


class DeepJetHLT(Classifier_base):

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
        "jet_sv_d3dsig"
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

<<<<<<< HEAD
        if attack is not None:
=======
            torch.save(
                {
                    "epoch": t,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss_train,
                    "acc_train": acc_train,
                    "loss_val": loss_val,
                    "acc_val": acc_val,
                },
                "{}/model_{}.pt".format(directory, t),
            )

            if np.mean(loss_validation) < best_loss_val:
                best_loss_val = np.mean(loss_validation)
                torch.save(
                    {
                        "epoch": t,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_train": loss_train,
                        "acc_train": acc_train,
                        "loss_val": loss_val,
                        "acc_val": acc_val,
                    },
                    "{}/best_model.pt".format(directory),
                )

        return (
            loss_train,
            loss_val,
            acc_train,
            acc_val,
        )

    def predict_model(self, dataloader, device, attack=None):
        self.eval()
        kinematics = []
        truths = []
        processes = []
        predictions = []
        for (
            global_features,
            cpf_features,
            npf_features,
            vtx_features,
            truth,
            weight,
            process,
        ) in tqdm(dataloader):
            # append jet_pt and jet_eta
            # this should be done differenlty in the future... avoid array slicing with magic numbers!
            torch.backends.cudnn.enabled = False
>>>>>>> master
            (
                glob,
                cpf,
                npf,
                vtx,
                truth,
            ) = attack(
                [
                    feature.float().to(device, non_blocking=True)
                    for feature in [
<<<<<<< HEAD
                        glob,
                        cpf,
                        npf,
                        vtx,
                        ]
                    ],
                truth.type(torch.LongTensor).to(device),
=======
                        global_features,
                        cpf_features,
                        npf_features,
                        vtx_features,
                    ]
                ],
                truth.type(torch.LongTensor).to(device, non_blocking=True),
>>>>>>> master
                self,
                loss_fn,
                device,
            ) 
        
        return (glob.detach(), cpf.detach(), npf.detach(), vtx.detach()), truth

    def __init__(self,
                 config,
                 cpf_conv = [64, 32, 32, 8],
                 vtx_conv = [64, 32, 32, 8],
                 n_layers_lstm = 1,
                 lstm_dim = [150, 50],
                 dense_clas_dim = [200, 100, 100, 100, 100, 100, 100, 100, 100],
                 **kwargs
        ):
        
        super().__init__(**kwargs)

<<<<<<< HEAD
        self.config = config

        _, cpf_dim, _, vtx_dim = self.all_model_features_length()  
        
        cpf_conv = [cpf_dim] + cpf_conv
        vtx_conv = [vtx_dim] + vtx_conv
        
        self.InputProcess = InputProcess_HLT(cpf_conv, vtx_conv)
=======
        predictions = np.concatenate(predictions)
        kinematics = np.concatenate(kinematics)
        truths = np.concatenate(truths)
        processes = np.concatenate(processes)
        return predictions, truths, kinematics, processes
>>>>>>> master

        dense_clas_dim_full = [sum(lstm_dim)] + dense_clas_dim
        self.DenseClassifier = DenseClassifier(dense_clas_dim_full)

<<<<<<< HEAD
        self.cpf_lstm = torch.nn.LSTM(
            input_size=cpf_conv[-1], hidden_size=lstm_dim[0], num_layers=n_layers_lstm, batch_first=True
        )
        self.vtx_lstm = torch.nn.LSTM(
            input_size=vtx_conv[-1], hidden_size=lstm_dim[1],  num_layers=n_layers_lstm, batch_first=True
        )

        self.cpf_bn = torch.nn.BatchNorm1d(lstm_dim[0], eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(lstm_dim[1], eps=0.001, momentum=0.6)
=======
        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Training...", total=dataloader.nits_expected)
            print("entering traing loop")
            for (
                global_features,
                cpf_features,
                npf_features,
                vtx_features,
                truth,
                weight,
                process,
            ) in dataloader:
                # We select either cuda float16 mixed precision or cpu float32 as LSTMs does not accept bfloat16
                with torch.autocast(
                    device_type=device, enabled=True if device == "cuda" else False
                ):
                    (
                        global_features,
                        cpf_features,
                        npf_features,
                        vtx_features,
                        truth,
                    ) = attack(
                        [
                            feature.float().to(device, non_blocking=True)
                            for feature in [
                                global_features,
                                cpf_features,
                                npf_features,
                                vtx_features,
                            ]
                        ],
                        truth.type(torch.LongTensor).to(device, non_blocking=True),
                        self,
                        self.loss_fn,
                    )

                    pred = self.forward(
                        global_features,
                        cpf_features,
                        npf_features,
                        vtx_features,
                    )
                    loss = self.loss_fn(pred, truth).mean()
>>>>>>> master

        self.cpf_dropout = nn.Dropout(0.1)
        self.vtx_dropout = nn.Dropout(0.1)

<<<<<<< HEAD
        self.Linear = nn.Linear(100, len(self.classes))
        
    def forward(self, inpt):
         
        _, cpf_features, _, vtx_features = inpt[0], inpt[1], inpt[2], inpt[3]
=======
                    losses.append(loss.item())
                    accuracy += (pred.argmax(1) == truth).type(torch.float).sum().item()
                    N += len(pred)
                    progress.update(
                        task, advance=1, description=f"Training...   | Loss: {loss:.2f}"
                    )
                    progress.columns[-1].text_format = "{}/{} its".format(
                        N // dataloader.batch_size,
                        (
                            "?"
                            if dataloader.nits_expected == len(dataloader)
                            else f"~{dataloader.nits_expected}"
                        ),
                    )
                progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Average loss: {np.array(losses).mean():.4f}")
        print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
        return losses, accuracy
>>>>>>> master

        #print(f"cpf_features.shape = {cpf_features.shape}")
        #print(f"vtx_features.shape = {vtx_features.shape}")
                
        cpf, vtx = self.InputProcess(cpf_features, vtx_features)
        
        cpf = self.cpf_lstm(torch.flip(cpf, dims=[1]))[0][:, -1]
        cpf = self.cpf_dropout(self.cpf_bn(cpf))

        vtx = self.vtx_lstm(torch.flip(vtx, dims=[1]))[0][:, -1]
        vtx = self.vtx_dropout(self.vtx_bn(vtx))

<<<<<<< HEAD
        fts = torch.cat((cpf, vtx), dim=1)
        fts = self.DenseClassifier(fts)

        output = self.Linear(fts)
=======
        with Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            TextColumn("0/? its"),
            expand=True,
        ) as progress:
            N = 0
            task = progress.add_task("Validation...", total=dataloader.nits_expected)
            for (
                global_features,
                cpf_features,
                npf_features,
                vtx_features,
                truth,
                weight,
                process,
            ) in dataloader:
                with torch.no_grad():
                    pred = self.forward(
                        *[
                            feature.float().to(device, non_blocking=True)
                            for feature in [
                                global_features,
                                cpf_features,
                                npf_features,
                                vtx_features,
                            ]
                        ]
                    )
                    loss = self.loss_fn(
                        pred, truth.type(torch.LongTensor).to(device, non_blocking=True)
                    ).mean()
                    losses.append(loss.item())

                    accuracy += (
                        (pred.argmax(1) == truth.to(device, non_blocking=True))
                        .type(torch.float)
                        .sum()
                        .item()
                    )
                    predictions = np.append(predictions, pred.to("cpu").numpy(), axis=0)
                    truths = np.append(truths, truth.to("cpu").numpy(), axis=0)
                    processes = np.append(processes, process.to("cpu").numpy(), axis=0)
                N += global_features.size(dim=0)
                progress.update(
                    task, advance=1, description=f"Validation... | Loss: {loss:.2f}"
                )
                progress.columns[-1].text_format = "{}/{} its".format(
                    N // dataloader.batch_size,
                    (
                        "?"
                        if dataloader.nits_expected == len(dataloader)
                        else f"~{dataloader.nits_expected}"
                    ),
                )
            progress.update(task, completed=dataloader.nits_expected)
        dataloader.nits_expected = N // dataloader.batch_size
        accuracy /= N
        print("  ", f"Validation loss: {np.array(losses).mean():.4f}")
        print("  ", f"Validation accuracy: {float(accuracy):.4f}")

        if verbose:
            print("Printing terminal ROC")
            terminal_roc(predictions, truths, title="Validation ROC")

        return losses, float(accuracy)
>>>>>>> master

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

class MoDJet(DeepJet):

    classes = {
        "b": ["isB"],
        "bb": ["isBB", "isGBB"],
        "leptonicB": ["isLeptonicB", "isLeptonicB_C"],
        "c": ["isC", "isCC", "isGCC"],
        "uds": ["isU", "isD", "isS"],
        "g": ["isG"],
<<<<<<< HEAD
    }
=======
    }

    global_features = [
        "jet_px",
        "jet_py",
        "jet_pz",
        "jet_energy",
        "jet_mass",
        "n_Cpfcand",
        "n_Npfcand",
        "nsv",
        "npv",
        "TagVarCSV_trackSumJetEtRatio",
        "TagVarCSV_trackSumJetDeltaR",
        "TagVarCSV_vertexCategory",
        "TagVarCSV_trackSip2dValAboveCharm",
        "TagVarCSV_trackSip2dSigAboveCharm",
        "TagVarCSV_trackSip3dValAboveCharm",
        "TagVarCSV_trackSip3dSigAboveCharm",
        "TagVarCSV_jetNSelectedTracks",
        "TagVarCSV_jetNTracksEtaRel",
    ]

    cpf_candidates = [
        "Cpfcan_px",
        "Cpfcan_py",
        "Cpfcan_pz",
        "Cpfcan_e",
        "Cpfcan_mass",
        "Cpfcan_BtagPf_trackEtaRel",
        "Cpfcan_BtagPf_trackPtRel",
        "Cpfcan_BtagPf_trackPPar",
        "Cpfcan_BtagPf_trackDeltaR",
        "Cpfcan_BtagPf_trackPParRatio",
        "Cpfcan_BtagPf_trackSip2dVal",
        "Cpfcan_BtagPf_trackSip2dSig",
        "Cpfcan_BtagPf_trackSip3dVal",
        "Cpfcan_BtagPf_trackSip3dSig",
        "Cpfcan_BtagPf_trackJetDistVal",
        "Cpfcan_ptrel",
        "Cpfcan_drminsv",
        "Cpfcan_VTX_ass",
        "Cpfcan_puppiw",
        "Cpfcan_chi2",
        "Cpfcan_quality",
    ]

    npf_candidates = [
        "Npfcan_px",
        "Npfcan_py",
        "Npfcan_pz",
        "Npfcan_e",
        "Npfcan_mass",
        "Npfcan_ptrel",
        "Npfcan_deltaR",
        "Npfcan_isGamma",
        "Npfcan_HadFrac",
        "Npfcan_drminsv",
        "Npfcan_puppiw",
    ]

    vtx_features = [
        "sv_px",
        "sv_py",
        "sv_pz",
        "sv_e",
        "sv_mass",
        "sv_deltaR",
        "sv_ntracks",
        "sv_chi2",
        "sv_normchi2",
        "sv_dxy",
        "sv_dxysig",
        "sv_d3d",
        "sv_d3dsig",
        "sv_costhetasvpv",
        "sv_enratio",
    ]

    feature_keys = [global_features, cpf_candidates, npf_candidates, vtx_features]

    def __init__(self, feature_edges=[18, 543, 818, 893], **kwargs):
        super(MoDJet, self).__init__(**kwargs)

        self.feature_edges = np.array(feature_edges)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        self.InputProcess = MoDInputProcess()
        self.DenseClassifier = MoDDenseClassifier()

        self.global_bn = torch.nn.BatchNorm1d(18, eps=0.001, momentum=0.6)
        self.cpf_lstm = torch.nn.LSTM(
            input_size=8, hidden_size=150, num_layers=1, batch_first=True
        )
        self.npf_lstm = torch.nn.LSTM(
            input_size=4, hidden_size=50, num_layers=1, batch_first=True
        )
        self.vtx_lstm = torch.nn.LSTM(
            input_size=8, hidden_size=50, num_layers=1, batch_first=True
        )

        self.cpf_bn = torch.nn.BatchNorm1d(150, eps=0.001, momentum=0.6)
        self.npf_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)
        self.vtx_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)

        self.cpf_dropout = torch.nn.Dropout(0.1)
        self.npf_dropout = torch.nn.Dropout(0.1)
        self.vtx_dropout = torch.nn.Dropout(0.1)

        self.Linear = torch.nn.Linear(100, len(self.classes))

        self.glob_integers = torch.tensor([5, 6, 7, 8, 11, 16, 17])
        self.cpf_integers = torch.tensor([17, 18, 19, 20])
        self.npf_integers = torch.tensor([7])
        self.vtx_integers = torch.tensor([6])
        self.integers = [
            self.glob_integers,
            self.cpf_integers,
            self.npf_integers,
            self.vtx_integers,
        ]
        self.glob_defaults = torch.tensor([0])
        self.cpf_defaults = torch.tensor([0])
        self.npf_defaults = torch.tensor([0])
        self.vtx_defaults = torch.tensor([0])
        self.defaults = [
            self.glob_defaults,
            self.cpf_defaults,
            self.npf_defaults,
            self.vtx_defaults,
        ]
>>>>>>> master
