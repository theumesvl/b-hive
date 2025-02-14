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

            return DeepJet(**kwargs)
        case ModelName.DeepJetHLT:
            from utils.models.deepjet import DeepJetHLT

            return DeepJetHLT(**kwargs)
        case ModelName.DeepJetTransformer:
            from utils.models.deepjettransformer import DeepJetTransformer

            return DeepJetTransformer(**kwargs)
        case ModelName.L1TKerasDeepSet:
            from utils.models.l1t_kerasDeepset import L1TKerasDeepSet

            return L1TKerasDeepSet(**kwargs)
        case ModelName.L1TTorchBase:
            from utils.models.l1t_base import L1TTorchBase

            return L1TTorchBase(**kwargs)
        case ModelName.MoDJet:
            from utils.models.deepjet import MoDJet

            return MoDJet(**kwargs)
        case ModelName.PAIReDTagger:
            from utils.models.PAIReDTagger import PAIReDTagger

            return PAIReDTagger(**kwargs)
        case ModelName.ParticleNet:
            from utils.models.particlenet_base import ParticleNetTagger

            return ParticleNetTagger(**kwargs)
        case ModelName.ParticleTransformer:
            from utils.models.particletransformer import ParticleTransformer

            return ParticleTransformer(**kwargs)
        case _:
            raise NotImplementedError
