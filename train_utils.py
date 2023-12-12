from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, CSVLogger
from model import VGG16, resnet18, resnet50
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics, bwt_metrics, class_accuracy_metrics
from torch.optim import SGD, Adam
from torch import cuda, flatten, stack
from torch import device as torch_device
from plot import training_acc_plot
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.checkpoint import maybe_load_checkpoint, save_checkpoint
import os
import SimCLR_models as simclr
from custom_plugins import EpochCheckpointing, EpochTesting
import torchvision.transforms as transforms
import torch
from slune.slune import get_csv_slog

def train(scenario, cl_strategy, name, device):
    """ Performs the training loop, checkpointints after each experience and each epoch."""
    fname = "checkpoints/"+name+".pkl"  # name of the checkpoint file
    cl_strategy, initial_exp = maybe_load_checkpoint(cl_strategy, fname, map_location=device) # load from checkpoint if exists
    cl_strategy.device = device
    # if checkpoint directory does not exist, create it
    directory = fname[0:[pos for pos, char in enumerate(fname) if char == "/"][-1]]
    if not os.path.exists(directory):
        os.makedirs(directory)
    # we add epoch checkpointing plugin to the strategy
    cl_strategy.plugins = cl_strategy.plugins + [EpochTesting(scenario.test_stream)] #TODO: could I do this in the constructor?
    print('Starting training...')
    # for experience in scenario.train_stream:
    for experience in scenario.train_stream[initial_exp:]:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # we train
        cl_strategy.train(experience)
        print('Training completed')

        # we checkpoint (save the model)
        save_checkpoint(cl_strategy, fname)
    print('Experiment completed')

def done_train_ssl(model, optimizer):
    """
    Once ssl is done we need to let the model know that it is now in classifier training mode, ie. we want to use classifier instead of training head and to freeze the encoder.
    We also reset the optimizer. 
    """
    model.set_train_classifier()
    if (type (optimizer).__name__ == 'SGD') and (optimizer.defaults['momentum'] != 0):
        optimizer = torch.optim.SGD(model.parameters(), lr=optimizer.lr, momentum=0.9)
    if (type (optimizer).__name__ == 'Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=optimizer.lr)

def tune_hyperparams(tune_type, data_name, model_name, optimizer_type, selection_metric='final_train_accuracy'):
    if selection_metric != 'final_train_accuracy':
        raise ValueError("Only final_train_accuracy is supported for hyperparameter tuning currently")
    slog = get_csv_slog()
    if data_name == 'SplitCIFAR10':
        data_name = 'CIFAR10'
    elif data_name == 'SplitCIFAR100':
        data_name = 'CIFAR100'
    else: 
        raise ValueError('Only support CIFAR10 and CIFAR100 for hyperparameter tuning currently')
    params = ['--tune_type='+tune_type, '--data_name='+data_name, '--model_name='+model_name, '--optimizer_type='+optimizer_type]
    params, value = slog.read(params, metric_name = selection_metric, select_by ='max')
    if tune_type == 'classification':
        lr = float([p for p in params if 'learning_rate' in p][0].split('=')[1])
        params = lr
    elif tune_type == 'ssl':
        lr = float([p for p in params if 'learning_rate' in p][0].split('=')[1])
        temperature = float([p for p in params if 'temperature' in p][0].split('=')[1])
        params = [lr, temperature]
    return params
