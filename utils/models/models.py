class ModelName:
    DeepJet = "DeepJet"
    DeepJetHLT = "DeepJetHLT"
    MoDJet = "MoDJet"
    ParticleTransformer = "ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"
    ParticleNet = "ParticleNet"
    ParticleNetHION = "ParticleNetHION"
    L1TKerasDeepSet = "L1TKerasDeepSet"
    L1TTorchBase = "L1TTorchBase"
    PAIReDTagger = "PAIReDTagger"


def BTaggingModels(model: str = "", *args, **kwargs):
    match model:
        case ModelName.DeepJet:
            from utils.models.deepjet import DeepJet
            return DeepJet(*args, **kwargs)
        case ModelName.DeepJetHLT:
            from utils.models.deepjet import DeepJetHLT
            return DeepJetHLT(*args, **kwargs)
        case ModelName.UParT_v0:
            from utils.models.UParT_v0 import UParT_v0
            return UParT_v0(*args, **kwargs)
        case ModelName.MoDJet:
            from utils.models.deepjet import MoDJet 
            return MoDJet(*args, **kwargs)
        case ModelName.ParticleTransformer:
            from utils.models.particletransformer import ParticleTransformer
            return ParticleTransformer(*args, **kwargs)
        case ModelName.FP16ParticleTransformer:
            from utils.models.fp16particletransformer import FP16ParticleTransformer
            return FP16ParticleTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformer:
            from utils.models.deepjettransformer import DeepJetTransformer
            return DeepJetTransformer(*args, **kwargs)
        case ModelName.ParticleNet_InPro:
            from utils.models.particlenet_InPro import ParticleNetTagger as ParticleNet_InPro
            return ParticleNet_InPro(*args, **kwargs)
        case ModelName.ParticleNet:
            from utils.models.particlenet_base import ParticleNetTagger
            return ParticleNetTagger(*args, **kwargs)
        case ModelName.L1TKerasDeepSet:
            from utils.models.l1t_kerasDeepset import L1TKerasDeepSet
            return L1TKerasDeepSet(*args, **kwargs)
        case ModelName.L1TTorchBase:
            from utils.models.l1t_base import L1TTorchBase
            return L1TTorchBase(*args, **kwargs)
        case ModelName.PAIReDTagger:
            from utils.models.PAIReDTagger import PAIReDTagger
            return PAIReDTagger(*args, **kwargs)
        case ModelName.LZ4PAIReDTagger:
            from utils.models.LZ4PAIReDTagger import LZ4PAIReDTagger
            return LZ4PAIReDTagger(*args, **kwargs)
        case _:
            raise NotImplementedError
