from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, CSVLogger
from model import VGG16, ResNet18, ResNet50
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics, forward_transfer_metrics, bwt_metrics, class_accuracy_metrics
from torch.optim import SGD, Adam
from torch import cuda
from plot import training_acc_plot
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.training.checkpoint import maybe_load_checkpoint, save_checkpoint
import os

def set_seed(seed):
    """Sets the seed for Python's `random`, NumPy, and PyTorch global generators"""
    RNGManager.set_random_seeds(seed)

def get_model(model_name, device, num_classes):
    """ Returns the model with the given name and device."""
    if model_name == "VGG16":
        model = VGG16(num_classes=num_classes)
    elif model_name == "ResNet18":
        model = ResNet18(num_classes=num_classes)
    elif model_name == "ResNet50":
        model = ResNet50(num_classes=num_classes)
    else:
        raise ValueError("Model not supported")
    model.to(device)
    return model

def get_eval_plugin(name="log", log_tensorboard=True, log_stdout=True, log_csv=False, log_text=False, track_classes=None):
    """ Returns an evaluation plugin with the desired loggers."""
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

def get_device(device):
    """ 
    If device False: sets device to cuda if available, otherwise cpu. 
    If device not false returns value.
    """
    if device == False:
        device = 'cuda' if cuda.is_available() else 'cpu'
    elif device in ["GPU", "gpu", "cuda", "CUDA"]:
        device = 'cuda'
    else:
        pass
    return device

def train_and_plot(scenario, cl_strategy, eval_plugin, name):
    """ Performs the training loop, supports checkpointing."""
    fname = "checkpoints/"+name+".pkl"  # name of the checkpoint file
    cl_strategy, initial_exp = maybe_load_checkpoint(cl_strategy, fname) # load from checkpoint if exists
    # if checkpoint directory does not exist, create it
    directory = fname[0:[pos for pos, char in enumerate(fname) if char == "/"][-1]]
    if not os.path.exists(directory):
        os.makedirs(directory)
    print('Starting fixed stratified stream experiment...')
    # for experience in scenario.train_stream:
    for experience in scenario.train_stream[initial_exp:]:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # we train
        cl_strategy.train(experience)
        print('Training completed')

        # we evaluate
        cl_strategy.eval(scenario.test_stream)
        print("Eval completed")

        # we checkpoint (save the model)
        save_checkpoint(cl_strategy, fname)
    print('Experiment completed')
    plot_results(eval_plugin, fname="plots/"+name+".png")
    print("Plotted Results")

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
