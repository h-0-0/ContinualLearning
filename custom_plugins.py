from avalanche.benchmarks.utils.data_loader import MultiDatasetDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer
from typing import Any
from torch import arange
from avalanche.training.checkpoint import save_checkpoint


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
        """ 
        Here we set the dataloader.
        We use a dataloader that can load samples from the new data and the buffer according to a split we define.
        If we run out of samples from the buffer we will oversample from it until we have finished sampling from the new data.
        """
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
        """ 
        We update the buffer after the experience.
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

    def resize(self, strategy: Any, new_size: int):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = self.buffer.subset(arange(self.max_size))

class EpochCheckpointing(SupervisedPlugin):
    def __init__(self, strategy, fname):
        """ 
        Replay plugin that allows you to specify the batch split to be used during replay, 
        ie. the number of samples to be used from the experience and from the buffer in each batch. 
        """
        super().__init__()
        self.strategy = strategy
        self.fname = fname

    def after_training_epoch(self, strategy: "BaseStrategy", **kwargs):
        """ 
        We checkpoint after each epoch.
        """
        save_checkpoint(self.cl_strategy, self.fname)