from avalanche.benchmarks.utils.data_loader import MultiDatasetDataLoader
from avalanche.core import SupervisedPlugin
from avalanche.training.storage_policy import ExemplarsBuffer
from typing import Any
from avalanche.benchmarks.utils import AvalancheDataset
from torch import arange
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.models import SimpleCNN
from torch.optim import Adam
from avalanche.training.supervised.strategy_wrappers import Naive
from torch.nn import CrossEntropyLoss


class FixedReplay(SupervisedPlugin):

    def __init__(self, storage_policy, buffer_data, max_size, bs1, bs2):
        """ A simple replay plugin with reservoir sampling. """
        super().__init__()
        self.storage_policy = storage_policy(max_size=max_size)
        self.storage_policy.update_from_dataset(buffer_data)
        self.bs1 = bs1
        self.bs2 = bs2

    def before_training_exp(self, strategy,
                            num_workers: int = 0, shuffle: bool = True,
                            **kwargs):
        """ Here we set the dataloader. We use a dataloader that can load samples from multiple datasets."""
        print("Override the dataloader.")
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
        print("Buffer update.")
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

    def update_from_dataset(self, new_data: AvalancheDataset):
        """Replaces buffer with the given dataset.

        :param new_data: the new dataset
        """
        self.buffer = new_data

    def resize(self, strategy: Any, new_size: int):
        """Update the maximum size of the buffer."""
        self.max_size = new_size
        if len(self.buffer) <= self.max_size:
            return
        self.buffer = self.buffer.subset(arange(self.max_size))
        
# GET DATA
scenario = SplitCIFAR10(n_experiences=5, shuffle=False)

# CREATE MODEL
model = SimpleCNN()

# CREATE OPTIMIZER
optimizer = Adam(model.parameters(), lr=0.0001)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, optimizer,
    criterion = CrossEntropyLoss(), train_epochs=1,
    train_mb_size = 128, eval_mb_size = 128
)

# TRAINING LOOP
print('Starting regular CL experiment...')
train_results = []
test_results = []
for experience in scenario.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    # train returns a dictionary which contains all the metric values
    train_results.append(cl_strategy.train(experience))
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    test_results.append(cl_strategy.eval(scenario.test_stream))
print('Experiment completed')