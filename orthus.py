from typing import Optional, Sequence
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate


class Orthus(SupervisedTemplate):
    """
    Orthus Strategy.
    Two headed model, one head is a projection network, the other is a classifier.
    We train the projection network on all data using a loss of your choice.
    We train the classifier on data from the buffer using a cross entropy loss, we assume data in the buffer iid.
    We identify the buffer examples by the task_id, if the task_id is -1 we assume it is a buffer example.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion_proj = CrossEntropyLoss(),
        criterion_class = CrossEntropyLoss(),
        class_inds_same : bool = False,
        **kwargs
    ):
        """
        :param model: an Avalanche model like the avalanche.models.SCRModel,
            where the train classifier uses a projection network (e.g., MLP)
            while the test classifier uses a NCM Classifier.
            Normalization should be applied between feature extractor
            and classifier.
        :param optimizer: PyTorch optimizer.
        :param criterion_proj: loss function for the projection network.
        :param criterion_class: loss function for the classifier.
        :param class_inds_same: 
            if True, whatever indices are currently used to indicate samples for classifier will be used for all subsequent minibatches, 
            if the models classifier_ind == None then we will update the indices regardless of the flag. 
            If False, the indices will be updated for each minibatch.
        """
        self.criterion_proj = criterion_proj
        self.criterion_class = criterion_class
        self.class_inds_same = class_inds_same
        super().__init__(
            model,
            optimizer,
            **kwargs
        )

    def _before_forward(self, **kwargs):
        """
        Updates the classifier_ind attribute of the model, so we only train the classifier on the buffer examples (or whatever is labeled as -1).
        If we know the same indices will be used for all subsequent minibatches (indicated by self.class_inds_same flag) we skip updating.
        """
        super()._before_forward(**kwargs)
        if self.model.classifier_ind == None or not self.class_inds_same:
            self.model.classifier_ind = [ind for ind, task_id in enumerate(self.mbatch[2]) if task_id == -1]
        else:
            pass

    def criterion(self):
        return self.criterion_proj(self.mb_output[0], self.mb_y) + self.criterion_class(self.mb_output[1], self.mb_y[self.model.classifier_ind])
