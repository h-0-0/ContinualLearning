from torch import cuda, flatten, stack
from torch.nn import CrossEntropyLoss
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer

from custom_plugins import BatchSplitReplay, FixedBuffer
import data as data
from utils import set_seed, get_eval_plugin, ssl_get_eval_plugin, get_optimizer, get_device, get_model, train, get_augmentations, done_train_ssl, tune_hyperparams
import self_supervised as ss

def regular(data_name, model_name, batch_size, learning_rate, epochs, n_tasks, device, optimizer_type, seed, early_stopping):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None:
        learning_rate = tune_hyperparams('classification', data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")

    # HANDLE DEVICE
    device = get_device(device)

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    name = data_name + "_" + str(n_tasks) + "_tasks" + "/" + model_name + "/" + optimizer_type +  "/regular/lr_" + str(learning_rate)
    print("Experiment name: " ,name)

    # GET DATA
    scenario = data.get_data(data_name, n_tasks=n_tasks, seed=seed)

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name, track_classes=[j for i in scenario.original_classes_in_exp for j in i])

    # SETUP OTHER PLUGINS
    if early_stopping > 0:
        plugins = [
            EarlyStoppingPlugin(early_stopping, "train_stream"),
            LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs))
        ]
    else:
        plugins = [
            LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs))
        ]

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
    train(scenario, cl_strategy, name, device)

def fixed_replay_stratify(data_name, model_name, batch_size, learning_rate, epochs, n_tasks, device, optimizer_type, seed, early_stopping, data2_name, batch_ratio, percentage):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None:
        learning_rate = tune_hyperparams('classification', data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    name = data_name + "_" + str(n_tasks) + "_tasks" + "/" + model_name + "/" + optimizer_type +  "/fixed_replay_stratify/percent_" + str(percentage) + "_ratio_" + str(batch_ratio) + "_lr_" + str(learning_rate)
    print("Experiment name: " ,name)

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    scenario, buffer_data = data.get_data(data_name, data2_name, n_tasks=n_tasks, strategy={"name":"stratify", "percentage":percentage}, seed=seed) 

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name, track_classes={0: [0,1,2,3,4,5,6,7,8,9], 2: [0,1,2,3,4,5,6,7,8,9]}) #TODO: make this automatic for dataset

    # SETUP OTHER PLUGINS
    # Construct batch sizes for benchmark and replay
    bs_bench = int(batch_size*batch_ratio)
    bs_replay = batch_size - bs_bench
    # Construct the plugins
    if early_stopping > 0:
        plugins =[
            BatchSplitReplay(FixedBuffer, buffer_data, max_size=len(buffer_data), bs1=bs_bench, bs2=bs_replay),
            LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)),
            EarlyStoppingPlugin(early_stopping, "train_stream")
        ]
    else:
        plugins = [
            BatchSplitReplay(FixedBuffer, buffer_data, max_size=len(buffer_data), bs1=bs_bench, bs2=bs_replay),
            LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs))
        ]

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
    train(scenario, cl_strategy, name, device)

def ssl(data_name, data2_name, model_name, ssl_batch_size, class_batch_size, learning_rate, ssl_epochs, class_epochs, n_tasks, device, optimizer_type, seed, early_stopping, temperature, replay):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None or temperature is None:
        tune_learning_rate, tune_temperature = tune_hyperparams('ssl', data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")
        if learning_rate is None:
            learning_rate = tune_learning_rate
        if temperature is None:
            temperature = tune_temperature

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    if replay > 0:
        name = data_name + "_" + str(n_tasks) + "_tasks" + "/" + model_name + "/" + optimizer_type +  "/ssl/lr_" + str(learning_rate) + "_temp_" + str(temperature) + "_reservoir_buffer_" + str(replay) + "_ssl"
    else:
        name = data_name + "_" + str(n_tasks) + "_tasks" + "/" + model_name + "/" + optimizer_type +  "/ssl/lr_" + str(learning_rate) + "_temp_" + str(temperature) + "_ssl"
    print("Experiment name: " ,name)

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    ssl_scenario = data.get_data(data_name, n_tasks=n_tasks, seed=seed, no_augmentation=True) # We turn off augmentations as we will be doing them as part of SSL
    class_scenario = data.get_data(data2_name, seed=seed)

    # CREATE MODEL
    num_classes = len([item for sublist in class_scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model("SimCLR_" + model_name, device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # SETUP OTHER PLUGINS
    # Construct the plugins
    plugins = [LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_epochs))]
    if early_stopping > 0:
        plugins.append(EarlyStoppingPlugin(early_stopping, "train_stream"))
    elif replay > 0:
        storage_policy = ReservoirSamplingBuffer(max_size=replay)
        replay_plugin = ReplayPlugin(
            mem_size=replay, storage_policy=storage_policy
        )
        plugins.append(replay_plugin)

    # DEFINE THE AUGMENTATIONS
    augs = get_augmentations(ssl_scenario)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = ssl_get_eval_plugin(name) 

    # PERFORM SSL
    # construct the strategy
    ssl_strategy = ss.SimCLR(
        model, optimizer,
        augmentations = augs, temperature = temperature,
        train_mb_size = ssl_batch_size, train_epochs = ssl_epochs, eval_mb_size = ssl_batch_size,
        evaluator = eval_plugin,
        device = device,
        plugins = plugins
    )
    # train
    train(ssl_scenario, ssl_strategy, name, device)
    done_train_ssl(model, optimizer) # absorb this into training strategy, is there after all experiences callback?

    # TRAIN CLASSIFIER
    name = name[:-4] + "_classification"
    eval_plugin = get_eval_plugin(name, track_classes=[j for i in class_scenario.original_classes_in_exp for j in i])
    class_strategy = Naive(
        model, optimizer,
        criterion = CrossEntropyLoss(), train_epochs = class_epochs,
        train_mb_size = class_batch_size, eval_mb_size = class_batch_size,
        evaluator = eval_plugin,
        device = device,
        plugins = [LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=class_epochs))]
    )
    # train
    train(class_scenario, class_strategy, name, device)  

# TODO: set the scipy RNG in set_seed so can remove the need for passing seed to get_data
# TODO: if you run experiment from checkpoint does logging continue from where it left off?