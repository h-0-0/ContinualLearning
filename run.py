from torch import cuda, flatten
from torch.nn import CrossEntropyLoss
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin

from custom_plugins import FixedReplay, FixedBuffer
import data as data
from param_tune import tune_hyperparams
from utils import get_eval_plugin, get_optimizer, get_device, plot_results, get_model


def regular(data_name, model_name, batch_size, learning_rate, epochs, load_model, save_model, n_tasks, device, optimizer_type, seed, early_stopping):
    # PERFORM/LOAD HYPERPARAMETER TUNING
    # tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="top_test_accuracy")

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario = data.get_data(data_name, n_tasks=n_tasks)

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

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
    tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="top_test_accuracy")

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario, buffer_data = data.get_data(data_name, data2_name, n_tasks=n_tasks, strategy={"name":"stratify", "percentage":percentage}) 

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

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