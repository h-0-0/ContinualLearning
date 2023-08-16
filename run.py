from torch import cuda, flatten
from torch.nn import CrossEntropyLoss
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin

from custom_plugins import BatchSplitReplay, FixedBuffer
import data as data
from param_tune import tune_hyperparams
from utils import set_seed, get_eval_plugin, get_optimizer, get_device, get_model, train_and_plot


def regular(data_name, model_name, batch_size, learning_rate, epochs, n_tasks, device, optimizer_type, seed, early_stopping):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None:
        learning_rate = tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")

    # HANDLE DEVICE
    device = get_device(device)

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    name = data_name + "/" + model_name + "/" + optimizer_type +  "/regular/lr_" + str(learning_rate)

    # GET DATA
    scenario = data.get_data(data_name, n_tasks=n_tasks, seed=seed)

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name="log/"+ name)

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
    train_and_plot(scenario, cl_strategy, eval_plugin, name)

def fixed_replay_stratify(data_name, model_name, batch_size, learning_rate, epochs, n_tasks, device, optimizer_type, seed, early_stopping, data2_name, batch_ratio, percentage):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None:
        learning_rate = tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    name = data_name + "/" + model_name + "/" + optimizer_type +  "/fixed_replay_stratify/percent_" + str(percentage) + "_ratio_" + str(batch_ratio) + "_lr_" + str(learning_rate)

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario, buffer_data = data.get_data(data_name, data2_name, n_tasks=n_tasks, strategy={"name":"stratify", "percentage":percentage}, seed=seed) 
    print(scenario)
    print(scenario.streams())
    return None
    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name="log/"+ name) 

    # SETUP OTHER PLUGINS
    # Construct batch sizes for benchmark and replay
    bs_bench = int(batch_size*batch_ratio)
    bs_replay = batch_size - bs_bench
    # Construct the plugins
    if early_stopping > 0:
        plugins =[
            BatchSplitReplay(FixedBuffer, buffer_data, max_size=len(buffer_data), bs1=bs_bench, bs2=bs_replay),
            EarlyStoppingPlugin(early_stopping, "train_stream")
        ]
    else:
        plugins = [
            BatchSplitReplay(FixedBuffer, buffer_data, max_size=len(buffer_data), bs1=bs_bench, bs2=bs_replay)
        ]

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # CREATE THE STRATEGY INSTANCE
    # Construct the strategy
    cl_strategy = SupervisedTemplate(
        model, optimizer,
        CrossEntropyLoss(), train_mb_size=batch_size, train_epochs=epochs, eval_mb_size=batch_size,
        evaluator=eval_plugin,
        device = device,
        plugins=plugins
    )

    # TRAINING LOOP
    train_and_plot(scenario, cl_strategy, eval_plugin, name)

# TODO: class specific metrics not recording correctly for stratify strategy, either fix or create custom metric
# TODO: add lr sheduling
# TODO: sort out plotting or remove it
# TODO: set the scipy RNG in set_seed so can remove the need for passing seed to get_data
# TODO: if you run experiment from checkpoint does logging continue from where it left off?

# CHECK: max_size
# CHECK: runs on correct device