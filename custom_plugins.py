from avalanche.benchmarks.utils.data_loader import MultiDatasetDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer


class FixedReplay(SupervisedPlugin):

    def __init__(self, mem_size, bs1, bs2):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.buffer = ExemplarsBuffer(max_size=mem_size)
        self.bs1 = bs1
        self.bs2 = bs2
    def before_training_exp(self, strategy: "SupervisedTemplate",
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Use a custom dataloader to combine samples from the current data and memory buffer. """
        strategy.dataloader = MultiDatasetDataLoader(
            datasets = [strategy.adapted_dataset, self.buffer.buffer] ,
            batch_sizes = [self.bs1, self.bs2],
            termination_dataset = 0,
            oversample_small_datasets = True,
            distributed_sampling = True,
            shuffle = shuffle,
            num_workers = num_workers,
        )