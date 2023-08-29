from typing import Optional, Sequence

import torch
from torch.nn import Module, functional
from torch.optim import Optimizer
from torchvision.transforms import Compose, Lambda

from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin


class NTXentLoss(torch.nn.Module):
    """
    NT-Xent loss as defined in the SimCLR paper:
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        """
        # TODO: edit this docstring
        Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        features: [bsz, n_views, f_dim]
        `n_views` is the number of crops from each image, better
        be L2 normalized in f_dim dimension

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = x.device

        # First we compute pairwise normalized similarities
        x = functional.normalize(x, dim=1)
        x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
        x_scale = x_scores / self.temperature   # scale with temperature

        # (2N-1)-way softmax without the score of i-th entry itself.
        # Set the diagonals to be large negative values, which become zeros after softmax.
        x_scale = x_scale - torch.eye(x_scale.size(0)).to(device) * 1e5

        # targets 2N elements.
        targets = torch.arange(x.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        return functional.cross_entropy(x_scale, targets.long().to(device))


class SimCLR(SupervisedTemplate):
    """
    # TODO: edit this docstring
    Supervised Contrastive Replay from https://arxiv.org/pdf/2103.13885.pdf.
    This strategy trains an encoder network in a self-supervised manner to
    cluster together examples of the same class while pushing away examples
    of different classes. It uses the Nearest Class Mean classifier on the
    embeddings produced by the encoder.

    Accuracy cannot be monitored during training (no NCM classifier).
    During training, NCRLoss is monitored, while during eval
    CrossEntropyLoss is monitored.

    The original paper uses an additional fine-tuning phase on the buffer
    at the end of each experience (called review trick, but not mentioned
    in the paper). This implementation does not implement the review trick.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        augmentations=Compose([Lambda(lambda el: el)]),
        mem_size: int = 100,
        temperature: int = 0.1,
        train_mb_size: int = 1,
        batch_size_mem: int = 100,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device="cpu",
        plugins: Optional[Sequence["BaseSGDPlugin"]] = None,
        evaluator=default_evaluator,
        eval_every=-1,
        peval_mode="epoch",
    ):
        """
        # TODO: edit this docstring
        :param model: an Avalanche model like the avalanche.models.SCRModel,
            where the train classifier uses a projection network (e.g., MLP)
            while the test classifier uses a NCM Classifier.
            Normalization should be applied between feature extractor
            and classifier.
        :param optimizer: PyTorch optimizer.
        :param augmentations: TorchVision Compose Transformations to augment
            the input minibatch. The augmented mini-batch will be concatenated
            to the original one (which includes the memory buffer).
            Note: only augmentations that can be applied to Tensors
            are supported.
        :param mem_size: replay memory size, used also at test time to
            compute class means.
        :param temperature: SCR Loss temperature.
        :param train_mb_size: mini-batch size for training. The default
            dataloader is a task-balanced dataloader that divides each
            mini-batch evenly between samples from all existing tasks in
            the dataset.
        :param batch_size_mem: number of examples drawn from the buffer.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param peval_mode: one of {'epoch', 'iteration'}. Decides whether the
            periodic evaluation during training should execute every
            `eval_every` epochs or iterations (Default='epoch').
        """

        self.augmentations = augmentations
        self.temperature = temperature

        self.loss_fun = NTXentLoss(temperature=self.temperature)

        super().__init__(
            model,
            optimizer,
            NTXentLoss(temperature=self.temperature),
            train_mb_size,
            train_epochs,
            eval_mb_size,
            device,
            plugins,
            evaluator,
            eval_every,
            peval_mode,
        )

    def criterion(self):
        return self.loss_fun(self.mb_output)

    def _before_forward(self, **kwargs):
        """
        Augment images for current mini-batch.
        """
        super()._before_forward(**kwargs)
        mb_x_augmented_1 = self.augmentations(self.mbatch[0])
        mb_x_augmented_2 = self.augmentations(self.mbatch[0])
        # We interleave augmented images
        mb_x_augmented = torch.cat([mb_x_augmented_1, mb_x_augmented_2], dim=0)
        n = mb_x_augmented_1.shape[0]
        indices = [[i, n+i] for i in range(n)]
        indices = [item for sublist in indices for item in sublist]
        mb_x_augmented_sorted = torch.zeros_like(mb_x_augmented)
        for i, ind in enumerate(indices):
            mb_x_augmented_sorted[i] = mb_x_augmented[ind]
        self.mbatch[0] = mb_x_augmented
        # TODO: make this more efficient