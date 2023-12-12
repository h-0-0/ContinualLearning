from typing import Optional, Sequence

import torch
from torch.nn import Module, functional
from torch.optim import Optimizer
from torchvision.transforms import Compose, Lambda

from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin


class GiveModelTask(SupervisedTemplate):
    """
    Gives the task id to the model as an additional input for the forward pass.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, **kwargs):
        """
        Forward pass of the model, but with the task id as an additional input.
        """
        task_id = torch.unique(self.mb_task_id).tolist()
        if len(task_id) > 1:
            raise Exception("More than one task id in minibatch")
        task_id = task_id[0]
        return self.model(x=self.mb_x, task_id=task_id)