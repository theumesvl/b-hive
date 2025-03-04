import luigi
from rich.console import Console

from utils.config.config_loader import ConfigLoader

class DatasetDependency(object):
    dataset_version = luigi.Parameter(
        default="dataset_version_01",
        description="Version Tag for dataset to save file with",
    )
    filelist = luigi.Parameter(
        description="txt file with input root files",
        significant=False,
        default="",
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.dataset_version,)
        return parts


class TestDatasetDependency(object):
    test_dataset_version = luigi.Parameter(
        default="dataset_version_01",
        description="Version Tag for dataset to save file with",
    )
    test_filelist = luigi.Parameter(
        description="txt file with input root files",
        significant=False,
        default="",
    )

    def store_parts(self):
        parts = super().store_parts()
        # append dataset-version to path
        parts += (self.test_dataset_version,)
        return parts


class TrainingDependency(object):
    training_version = luigi.Parameter(
        default="training_version_01",
        description="Version Tag for training to save file with",
    )
    epochs = luigi.IntParameter(default=1)
    model_name = luigi.Parameter()
    n_threads = luigi.IntParameter(
        default=4, description="Number of threads to use for dataloader."
    )
    batch_size = luigi.IntParameter(default=1000)
    learning_rate = luigi.FloatParameter(default=0.0001)

    def store_parts(self):
        parts = super().store_parts()

        parts += (self.training_version,)
        parts += (self.model_name,)
        parts += ("epochs_{0:d}".format(self.epochs),)

        return parts

    def set_device(self):
        c = Console()
        from utils.models.models import BTaggingModels
        from torch.nn import Module
        if issubclass(type(BTaggingModels(self.model_name, ConfigLoader.load_config(self.config))), Module):
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                c.print(
                    "[black on yellow]Warning:", "CUDA device available. Running on cuda!"
                )
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"
            c.print(
                "[black on yellow]Warning:", "No CUDA device available. Running on cpu..."
            )


class AttackDependency(object):

    attack = luigi.Parameter(
        default="nominal", description="Specify adversarial attack to use."
    )
    attack_magnitude = luigi.FloatParameter(
        default=0.0,
        description="Only use in combination with attack!=nominal. Set the magnitude for choosen attack.",
    )
    attack_iterations = luigi.IntParameter(
        default=1,
        description="Only use in combination with attack!=None and attack_magnitude!=0. Set the number of interations for choosen attack, if applicable.",
    )
    attack_individual_factors = luigi.BoolParameter(
        default=False,
        description="Decides whether individual attack magnitudes should be used per feature or not.",
    )
    attack_reduce = luigi.BoolParameter(
        default=True,
        description="Decides whether default values and integer values should be changed or not.",
    )
    attack_restrict_impact = luigi.FloatParameter(
        default=-1.0,
        description="Sets a maximal l-inf distance that each feature can be changed as a fraction of the nominal one. -1.0 means no restriction.",
    )
    attack_overshoot = luigi.FloatParameter(
        default=0.02,
        description="Only use in combination with attack==jetfool. Used to prevent vanishing updates.",
    )
    attack_uncertainty = luigi.FloatParameter(
        default=1.0,
        description="Only use in combination with attack==optimizer. Sets the allowed and scaled global uncertainty for input features.",
    )

    def store_parts(self):
        parts = super().store_parts()

        parts += (self.attack,)
        if self.attack_magnitude > 0.0:
            parts += ("epsilon_{}".format(self.attack_magnitude),)
            parts += ("iterations_{}".format(self.attack_iterations),)
        if self.attack == "jetfool":
            parts += ("overshoot_{}".format(self.attack_overshoot),)
        if self.attack == "optimizer":
            parts += ("uncertainty_{}".format(self.attack_uncertainty),)

        return parts


class TestAttackDependency(object):

    test_attack = luigi.Parameter(
        default="nominal", description="Specify adversarial attack to use for testing."
    )
    test_attack_magnitude = luigi.FloatParameter(
        default=0.0,
        description="Only use in combination with attack!=nominal. Set the magnitude for choosen attack for testing.",
    )
    test_attack_iterations = luigi.IntParameter(
        default=1,
        description="Only use in combination with attack!=None and attack_magnitude!=0. Set the number of interations for choosen attack, if applicable, for testing.",
    )
    test_attack_individual_factors = luigi.BoolParameter(
        default=False,
        description="Decides whether individual attack magnitudes should be used per feature or not, for testing.",
    )
    test_attack_reduce = luigi.BoolParameter(
        default=True,
        description="Decides whether default values and integer values should be changed or not, for testing.",
    )
    test_attack_restrict_impact = luigi.FloatParameter(
        default=-1.0,
        description="Sets a maximal l-inf distance that each feature can be changed as a fraction of the nominal one. -1.0 means no restriction, for testing.",
    )
    test_attack_overshoot = luigi.FloatParameter(
        default=0.02,
        description="Only use in combination with attack==jetfool. Used to prevent vanishing updates.",
    )
    test_attack_uncertainty = luigi.FloatParameter(
        default=1.0,
        description="Only use in combination with attack==optimizer. Sets the allowed and scaled global uncertainty for input features.",
    )

    def store_parts(self):
        parts = super().store_parts()

        parts += (f"test_attack_{self.test_attack}",)
        if self.test_attack_magnitude > 0.0:
            parts += ("test_epsilon_{}".format(self.test_attack_magnitude),)
            parts += ("test_iterations_{}".format(self.test_attack_iterations),)
        if self.test_attack == "jetfool":
            parts += ("test_overshoot_{}".format(self.test_attack_overshoot),)
        if self.test_attack == "optimizer":
            parts += ("uncertainty_{}".format(self.test_attack_uncertainty),)

        return parts
