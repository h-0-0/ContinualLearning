from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, CSVLogger
from model import VGG16, ResNet18, ResNet50
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

def set_seed(seed):
    """Sets the seed for Python's `random`, NumPy, and PyTorch global generators"""
    RNGManager.set_random_seeds(seed)

def get_model(model_name, device, num_classes):
    """ Returns the model with the given name and device."""
    if model_name == "VGG16":
        model = VGG16(num_classes)
    elif model_name == "ResNet18":
        model = ResNet18(num_classes)
    elif model_name == "ResNet50":
        model = ResNet50(num_classes)
    elif model_name ==  "SimCLR_VGG16":
        model = simclr.VGG16(num_classes)
    elif model_name ==  "SimCLR_ResNet18":
        model = simclr.ResNet18(num_classes)
    elif model_name ==  "SimCLR_ResNet50":
        model = simclr.ResNet50(num_classes)
    else:
        raise ValueError("Model not supported")
    model.to(device)
    return model

def get_eval_plugin(name, log_tensorboard=True, log_stdout=True, log_csv=False, log_text=False, track_classes=None):
    """ Returns an evaluation plugin with the desired loggers."""
    name = "log/" + name
    loggers = []
    if log_tensorboard:
        tb_logger = TensorboardLogger(tb_log_dir=name)
        loggers.append(tb_logger)
    if log_text:
        text_logger = TextLogger(open(name+'.txt', 'a'))
        loggers.append(text_logger)
    if log_stdout:
        interactive_logger = InteractiveLogger()
        loggers.append(interactive_logger)
    if log_csv:
        csv_logger = CSVLogger(name+'.csv')
        loggers.append(csv_logger)

    eval_plugin = EvaluationPlugin(
        # Metrics that use training stream
        accuracy_metrics(minibatch=True, epoch=True, stream=True), 
        loss_metrics(minibatch=True, epoch=True, stream=True), 
        class_accuracy_metrics(classes=track_classes, minibatch=True, epoch=True, stream=True),
        # Metrics that use evaluation stream
        # forward_transfer_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        # Other metrics
        cpu_usage_metrics(experience=True),
        disk_usage_metrics(experience=True),
        timing_metrics(epoch=True, epoch_running=True),
        loggers=loggers
    )
    return eval_plugin 

def ssl_get_eval_plugin(name, log_tensorboard=True, log_stdout=True, log_csv=False, log_text=False):
    """ Returns an evaluation plugin with the desired loggers."""
    name = "log/" + name
    loggers = []
    if log_tensorboard:
        tb_logger = TensorboardLogger(tb_log_dir=name)
        loggers.append(tb_logger)
    if log_text:
        text_logger = TextLogger(open(name+'.txt', 'a'))
        loggers.append(text_logger)
    if log_stdout:
        interactive_logger = InteractiveLogger()
        loggers.append(interactive_logger)
    if log_csv:
        csv_logger = CSVLogger(name+'.csv')
        loggers.append(csv_logger)

    eval_plugin = EvaluationPlugin(
        # Metrics that use training stream
        loss_metrics(minibatch=True, epoch=True, stream=True), 
        # Other metrics
        cpu_usage_metrics(experience=True),
        disk_usage_metrics(experience=True),
        timing_metrics(epoch=True, epoch_running=True),
        loggers=loggers
    )
    return eval_plugin 

def get_optimizer(optimizer_type, model, learning_rate):
    """ Returns the optimizer with the desired type."""
    if optimizer_type == "SGD":
        optimizer = SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD_momentum":
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError("Optimizer not supported")
    return optimizer

def get_device(device = None):
    """ 
    If device False: sets device to cuda if available, otherwise cpu. 
    If device not false returns value.
    """
    if device is None:
        device = torch_device('cuda') if cuda.is_available() else torch_device('cpu')
    elif device in ["GPU", "gpu", "cuda", "CUDA"]:
        device = torch_device('cuda')
    elif(device in ["CPU", "cpu"]):
        device = torch_device('cpu')
    else:
        raise ValueError("Device not supported")
    print("Using device: ", device)
    return device

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
    cl_strategy.plugins = cl_strategy.plugins + [EpochCheckpointing(cl_strategy, fname), EpochTesting(scenario.test_stream)] #TODO: could I do this in the constructor?
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

def plot_results(eval_plugin, fname=None):
    """ Plots the results from the metrics."""
    all_metrics = eval_plugin.get_all_metrics()
    fig = training_acc_plot(all_metrics)
    # Save the figure
    if fname is not None:
        directory = fname[:-len(fname[-len(fname.split(os.path.sep)[-1])])]
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(fname)
    else:
        fig.savefig("plots/"+"training_acc_plot.png")

def get_augmentations(scenario):
    """
    Returns composition of augmentations for self-supervised learning.
    """
    # we define lambda functions for the augmentations
    _, og_height, og_width = scenario.original_train_dataset[0][0].shape
    random_resized_crop_lambda_twice_per_image = transforms.Lambda(
        lambda imgs: 
            stack(
                [transforms.RandomResizedCrop(size=(og_height, og_width), scale=(0.2, 0.8), antialias=True)(img) for img in imgs for _ in range(2)]
            )
    )
    color_distort_lambda = transforms.Lambda(
        lambda imgs: 
            stack(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img) for img in imgs]
            )
    )
    # we create a composition of transformations including the lambda functions
    augmentations = transforms.Compose([
        random_resized_crop_lambda_twice_per_image,    # Apply random cropping and resizing to original size
        color_distort_lambda           # Apply color distortion
    ]) 
    return augmentations

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