from torch import cuda as cuda
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation.plot_utils import learning_curves_plot

from model import VGG16
from custom_plugins import FixedReplay, FixedBuffer
import data as data
import matplotlib.pyplot as plt

def get_model(model_name, device):
    """ Returns the model with the given name and device."""
    if model_name == "VGG16":
        model = VGG16()
    else:
        raise ValueError("Model not supported")
    model.to(device)
    return model

def get_eval_plugin(log_tensorboard=True, log_text=False, log_stdout=True):
    """ Returns an evaluation plugin with the desired loggers."""
    if log_tensorboard:
        tb_logger = TensorboardLogger()
    if log_text:
        text_logger = TextLogger(open('log.txt', 'a'))
    if log_stdout:
        interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
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
    fig = learning_curves_plot(all_metrics)
    # Save the figure
    if name is not None:
        fig.savefig(name)
    else:
        fig.savefig('learning_curves.png')


def regular(data_name, model_name, batch_size, learning_rate, epochs, load_model, save_model, n_tasks, device, optimizer_type, seed):
    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario = data.get_data(data_name, n_tasks)

    # CREATE MODEL
    model = get_model(model_name, device)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin()

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = SupervisedTemplate(
        model, optimizer,
        criterion = CrossEntropyLoss(), train_epochs=epochs,
        train_mb_size = batch_size, eval_mb_size = batch_size,
        evaluator=eval_plugin,
        device = device
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
    plot_results(eval_plugin, name="regular.png")
    print("Plotted Results")
    # TODO: sort naming

def fixed_replay_stratify(data_name, model_name, batch_size, learning_rate, epochs, load_model, save_model, n_tasks, device, optimizer_type, seed, data2_name, batch_ratio, percentage):
    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario, buffer_data = data.get_data(data_name, data2_name, n_tasks=n_tasks, strategy={"name":"stratify", "percentage":percentage}) 

    # CREATE MODEL
    model = get_model(model_name, device)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin()

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    # Construct batch sizes for benchmark and replay
    bs_bench = int(batch_size*batch_ratio)
    bs_replay = batch_size - bs_bench
    # Construct the strategy
    cl_strategy = SupervisedTemplate(
        model, optimizer,
        CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=epochs, eval_mb_size=batch_size,
        evaluator=eval_plugin,
        device = device,
        plugins=[FixedReplay(FixedBuffer, buffer_data, max_size=200, bs1=bs_bench, bs2=bs_replay)]
    )

    # TRAINING LOOP
    print('Starting fixed stratified stream experiment...')
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
    plot_results(eval_plugin, name="fixed_replay_stratify.png")
    print("Plotted Results")
    # TODO: sort naming

# TODO: implement load, save and checkpointing
# TODO: implement seed
# TODO: add early stopping
# TODO: add choosing hyperparameter

# CHECK: runs on correct device
# CHECK: stratify split 