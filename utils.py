from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, CSVLogger
from model import VGG16
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics, forward_transfer_metrics, bwt_metrics
from torch.optim import SGD, Adam
from torch import cuda
from plot import training_acc_plot

def get_model(model_name, device):
    """ Returns the model with the given name and device."""
    if model_name == "VGG16":
        model = VGG16()
    else:
        raise ValueError("Model not supported")
    model.to(device)
    return model

def get_eval_plugin(name="log", log_tensorboard=True, log_stdout=True, log_csv=False, log_text=False):
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
        accuracy_metrics(minibatch=True, epoch=True),
        loss_metrics(minibatch=True, epoch=True),
        # Metrics that use evaluation stream
        forward_transfer_metrics(experience=True, stream=True),
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

def plot_results(eval_plugin, name=None):
    """ Plots the results from the metrics."""
    all_metrics = eval_plugin.get_all_metrics()
    fig = training_acc_plot(all_metrics)
    # Save the figure
    if name is not None:
        fig.savefig(name)
    else:
        fig.savefig("plots/"+"training_acc_plot.png")
