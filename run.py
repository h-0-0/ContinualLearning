from torch import cuda as cuda
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, disk_usage_metrics, forward_transfer_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin

from model import VGG16
from custom_plugins import FixedReplay, FixedBuffer
import data as data
from plot import training_acc_plot
from param_tune import tune_hyperparameters


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


def regular(data_name, model_name, batch_size, learning_rate, epochs, load_model, save_model, n_tasks, device, optimizer_type, seed, early_stopping):
    # PERFORM/LOAD HYPERPARAMETER TUNING
    tune_hyperparameters(data_name, model_name, optimizer_type, selection_metric="top_test_accuracy")

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario = data.get_data(data_name, n_tasks=n_tasks)

    # CREATE MODEL
    model = get_model(model_name, device)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name="log/"+ data_name + "/" + model_name + "/" +"regular")

    # SETUP OTHER PLUGINS
    if early_stopping > 0:
        plugins = [EarlyStoppingPlugin(early_stopping, "train_stream")]
    else:
        plugins = []

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model, optimizer,
        criterion = CrossEntropyLoss(), train_epochs=epochs,
        train_mb_size = batch_size, eval_mb_size = batch_size,
        evaluator=eval_plugin,
        device = device,
        plugins=plugins
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
    print("train_results: ", train_results) #TODO: remove
    print("test_results: ", test_results) #TODO: remove
    print('Experiment completed')
    plot_results(eval_plugin, name="regular.png")
    print("Plotted Results")

def fixed_replay_stratify(data_name, model_name, batch_size, learning_rate, epochs, load_model, save_model, n_tasks, device, optimizer_type, seed, early_stopping, data2_name, batch_ratio, percentage):
    # PERFORM/LOAD HYPERPARAMETER TUNING
    tune_hyperparameters(data_name, model_name, optimizer_type, selection_metric="top_test_accuracy")

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario, buffer_data = data.get_data(data_name, data2_name, n_tasks=n_tasks, strategy={"name":"stratify", "percentage":percentage}) 

    # CREATE MODEL
    model = get_model(model_name, device)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name="log/"+ data_name + "/" + model_name + "/" +"fixed_replay_stratify") 

    # SETUP OTHER PLUGINS
    print("Length of buffer data: ", len(buffer_data)) #TODO: remove
    if early_stopping > 0:
        plugins =[
            FixedReplay(FixedBuffer, buffer_data, max_size=len(buffer_data), bs1=bs_bench, bs2=bs_replay),
            EarlyStoppingPlugin(early_stopping, "train_stream")
        ]
    else:
        plugins = [
            FixedReplay(FixedBuffer, buffer_data, max_size=len(buffer_data), bs1=bs_bench, bs2=bs_replay)
        ]

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # CREATE THE STRATEGY INSTANCE
    # Construct batch sizes for benchmark and replay
    bs_bench = int(batch_size*batch_ratio)
    bs_replay = batch_size - bs_bench
    # Construct the strategy
    cl_strategy = SupervisedTemplate(
        model, optimizer,
        CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=epochs, eval_mb_size=batch_size,
        evaluator=eval_plugin,
        device = device,
        plugins=plugins
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


# TODO: do I need to collect train and test results?
# TODO: implement seed
# TODO: implement load, save and checkpointing
# TODO: reorganise directory structure to better organize code
# TODO: sort out plotting or remove it

# CHECK: max_size
# CHECK: runs on correct device
# CHECK: hyperparameter tuning
# CHECK: early stopping
# CHECK: stratify split 