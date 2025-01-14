from utils.models.particletransformer import ParticleTransformer, ParticleTransformerHLT, GlobalParticleTransformerHLT
from utils.models.UParT_v0 import UParT_v0
from utils.models.fp16particletransformer import FP16ParticleTransformer
from utils.models.deepjettransformer import DeepJetTransformer, DeepJetTransformerHLT, GlobalDeepJetTransformerHLT
#from utils.models.particlenet_base import ParticleNetTagger
from utils.models.particlenet_InPro import ParticleNetTagger as ParticleNet_InPro
from utils.models.particlenet_InPro import ParticleNetTaggerHLT as ParticleNet_InProHLT
from utils.models.deepjet import DeepJet, MoDJet, DeepJetHLT
from utils.models.l1t_base import L1TTorchBase
from utils.models.PAIReDTagger import PAIReDTagger
from utils.models.LZ4PAIReDTagger import LZ4PAIReDTagger


class ModelName:
    DeepJet = "DeepJet"
    DeepJetHLT = "DeepJetHLT"
    MoDJet = "MoDJet"
    ParticleTransformer = "ParticleTransformer"
    ParticleTransformerHLT = "ParticleTransformerHLT"
    GlobalParticleTransformerHLT = "GlobalParticleTransformerHLT"
    UParT_v0 = "UParT_v0"
    FP16ParticleTransformer = "FP16ParticleTransformer"
    DeepJetTransformer = "DeepJetTransformer"
    DeepJetTransformerHLT = "DeepJetTransformerHLT"
    GlobalDeepJetTransformerHLT = "GlobalDeepJetTransformerHLT"
    #ParticleNet = "ParticleNet"
    ParticleNet_InPro = "ParticleNet_InPro"
    ParticleNet_InProHLT = "ParticleNet_InProHLT"
    ParticleNetHION = "ParticleNetHION"
    L1TKerasDeepSet = "L1TKerasDeepSet"
    L1TTorchBase = "L1TTorchBase"
    PAIReDTagger = "PAIReDTagger"
    LZ4PAIReDTagger = "LZ4PAIReDTagger"

    
def BTaggingModels(model: str = "", *args, **kwargs):
    match model:
        case ModelName.DeepJet:
            return DeepJet(*args, **kwargs)
        case ModelName.DeepJetHLT:
            return DeepJetHLT(*args, **kwargs)
        case ModelName.UParT_v0:
            return UParT_v0(*args, **kwargs)
        case ModelName.MoDJet:
            return MoDJet(*args, **kwargs)
        case ModelName.ParticleTransformer:
            return ParticleTransformer(*args, **kwargs)
        case ModelName.ParticleTransformerHLT:
            return ParticleTransformerHLT(*args, **kwargs)
        case ModelName.GlobalParticleTransformerHLT:
            return GlobalParticleTransformerHLT(*args, **kwargs)
        case ModelName.FP16ParticleTransformer:
            return FP16ParticleTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformer:
            return DeepJetTransformer(*args, **kwargs)
        case ModelName.DeepJetTransformerHLT:
            return DeepJetTransformerHLT(*args, **kwargs)
        case ModelName.GlobalDeepJetTransformerHLT:
            return GlobalDeepJetTransformerHLT(*args, **kwargs)
        case ModelName.ParticleNet_InPro:
            return ParticleNet_InPro(*args, **kwargs)
        case ModelName.ParticleNet_InProHLT:
            return ParticleNet_InProHLT(*args, **kwargs)
        #case ModelName.ParticleNet:
        #    return ParticleNetTagger(*args, **kwargs)
        case ModelName.L1TKerasDeepSet:
            from utils.models.l1t_kerasDeepset import L1TKerasDeepSet
            return L1TKerasDeepSet(*args, **kwargs)
        case ModelName.L1TTorchBase:
            return L1TTorchBase(*args, **kwargs)
        case ModelName.PAIReDTagger:
            return PAIReDTagger(*args, **kwargs)
        case ModelName.LZ4PAIReDTagger:
            return LZ4PAIReDTagger(*args, **kwargs)
        case _:
            raise NotImplementedError(f"The model '{model}' is not implemented.")
