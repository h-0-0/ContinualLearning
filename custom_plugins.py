from avalanche.benchmarks.utils.data_loader import MultiDatasetDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer
from typing import Any
from avalanche.benchmarks.utils import AvalancheDataset
from torch import arange


class BatchSplitReplay(SupervisedPlugin):

    def __init__(self, storage_policy, buffer_data, max_size, bs1, bs2):
        """ 
        Replay plugin that allows you to specify the batch split to be used during replay, 
        ie. the number of samples to be used from the experience and from the buffer in each batch. 
        """
        super().__init__()
        self.storage_policy = storage_policy(max_size=max_size)
        self.storage_policy.buffer = buffer_data
        self.bs1 = bs1
        self.bs2 = bs2

    def before_training_exp(self, strategy,
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Here we set the dataloader. We use a dataloader that can load samples from multiple datasets."""
        strategy.dataloader = MultiDatasetDataLoader(
            datasets = [strategy.adapted_dataset, self.storage_policy.buffer] ,
            batch_sizes = [self.bs1, self.bs2],
            termination_dataset = 0,
            oversample_small_datasets = True,
            distributed_sampling = True,
            shuffle = shuffle,
            num_workers = num_workers,
        )

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        """ We update the buffer after the experience.
            You can use a different callback to update the buffer in a different place
        """
        self.storage_policy.update(strategy, **kwargs)

class FixedBuffer(ExemplarsBuffer):
    def __init__(self, max_size: int):
        """
        A buffer for replay that does not modify istelf unless explicitly.
        """
        super().__init__(max_size)

    def update(self, strategy: Any, **kwargs):
        """Update buffer. We wish not to update the buffer automatically."""
        pass

    # def update_from_dataset(self, new_data: AvalancheDataset):
    #     """Replaces buffer with the given dataset.

    #     :param new_data: the new dataset
    #     """
    #     self.buffer = new_data

    def resize(self, strategy: Any, new_size: int):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = self.buffer.subset(arange(self.max_size))
