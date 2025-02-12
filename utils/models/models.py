class ModelName:
    DeepJet = "DeepJet"
    DeepJetHLT = "DeepJetHLT"
    DeepJetTransformer = "DeepJetTransformer"
    L1TKerasDeepSet = "L1TKerasDeepSet"
    L1TTorchBase = "L1TTorchBase"
    MoDJet = "MoDJet"
    PAIReDTagger = "PAIReDTagger"
    ParticleNet = "ParticleNet"
    ParticleTransformer = "ParticleTransformer"


def BTaggingModels(model: str = "", *args, **kwargs):
    match model:
        case ModelName.DeepJet:
            from utils.models.deepjet import DeepJet

            return DeepJet(*args, **kwargs)
        case ModelName.DeepJetHLT:
            from utils.models.deepjet import DeepJetHLT

            return DeepJetHLT(*args, **kwargs)
        case ModelName.DeepJetTransformer:
            from utils.models.deepjettransformer import DeepJetTransformer

            return DeepJetTransformer(*args, **kwargs)
        case ModelName.L1TKerasDeepSet:
            from utils.models.l1t_kerasDeepset import L1TKerasDeepSet

            return L1TKerasDeepSet(*args, **kwargs)
        case ModelName.L1TTorchBase:
            from utils.models.l1t_base import L1TTorchBase

            return L1TTorchBase(*args, **kwargs)
        case ModelName.MoDJet:
            from utils.models.deepjet import MoDJet

            return MoDJet(*args, **kwargs)
        case ModelName.PAIReDTagger:
            from utils.models.PAIReDTagger import PAIReDTagger

            return PAIReDTagger(*args, **kwargs)
        case ModelName.ParticleNet:
            from utils.models.particlenet_base import ParticleNetTagger

            return ParticleNetTagger(*args, **kwargs)
        case ModelName.ParticleTransformer:
            from utils.models.particletransformer import ParticleTransformer

            return ParticleTransformer(*args, **kwargs)
        case _:
            raise NotImplementedError
