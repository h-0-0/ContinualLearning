from typing import Optional, Sequence

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torchvision.transforms import Compose, Lambda

from avalanche.core import BaseSGDPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate

class NTXentLoss(torch.nn.Module):
    """
    NT-Xent loss as defined in the SimCLR paper:
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
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
        device = features.device
        # First we compute the pairwise cosine similarity matrix 
        sim_mat = torch.nn.functional.cosine_similarity(
            features.unsqueeze(1), features.unsqueeze(0), dim=-1
        )
        # Mask out cosine similarity to itself
        self_mask = torch.eye(sim_mat.shape[0], dtype=torch.bool, device=device)
        sim_mat.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=sim_mat.shape[0] // 2, dims=0)
        # InfoNCE loss
        sim_mat = sim_mat / self.temperature
        nll = -sim_mat[pos_mask] + torch.logsumexp(sim_mat, dim=-1)
        nll = nll.mean()

        return nll


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

        self.pre_train_loss = NTXentLoss(temperature=self.temperature)
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.eval_loss = torch.nn.CrossEntropyLoss()

        self.is_pretraining = True

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
        if self.is_pretraining:
            return self.pre_train_loss(self.mb_output)
        elif self.is_training:
            return self.train_loss(self.mb_output, self.mb_y)
        else:
            return self.eval_loss(self.mb_output, self.mb_y)

    def _before_forward(self, **kwargs):
        """
        Augment images for current mini-batch.
        """
        # TODO: this bit correct?
        assert self.is_pretraining
        super()._before_forward(**kwargs)
        mb_x_augmented = self.augmentations(self.mbatch[0])
        self.mbatch[0] = mb_x_augmented
    
    def done_pretraining(self, new_batch_size = None, new_epochs = None):
        self.is_pretraining = False
        self.model.pretraining = False
        self.train_mb_size = new_batch_size if new_batch_size is not None else self.train_mb_size
        self.train_epochs = new_epochs if new_epochs is not None else self.train_epochs