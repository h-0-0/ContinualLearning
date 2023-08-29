from torch import cuda, flatten, stack
from torch.nn import CrossEntropyLoss
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin

from custom_plugins import BatchSplitReplay, FixedBuffer
import data as data
from param_tune import tune_hyperparams, ssl_tune_hyperparams
from utils import set_seed, get_eval_plugin, ssl_get_eval_plugin, get_optimizer, get_device, get_model, train, pretrain
import self_supervised as ss

def regular(data_name, model_name, batch_size, learning_rate, epochs, n_tasks, device, optimizer_type, seed, early_stopping):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None:
        learning_rate = tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")

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

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name="log/"+ name, track_classes=[j for i in scenario.original_classes_in_exp for j in i])

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
    train(scenario, cl_strategy, name, device)

def fixed_replay_stratify(data_name, model_name, batch_size, learning_rate, epochs, n_tasks, device, optimizer_type, seed, early_stopping, data2_name, batch_ratio, percentage):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None:
        learning_rate = tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")

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

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name="log/"+ name, track_classes=[j for i in scenario.original_classes_in_exp for j in i]) 

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
    train(scenario, cl_strategy, name, device)

def ssl(data_name, data2_name, model_name, ssl_batch_size, class_batch_size, learning_rate, ssl_epochs, class_epochs, n_tasks, device, optimizer_type, seed, early_stopping, temperature):
    # SET THE SEED 
    set_seed(seed)

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if learning_rate is None or temperature is None:
        tune_learning_rate, tune_temperature = ssl_tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="final_train_accuracy")
        if learning_rate is None:
            learning_rate = tune_learning_rate
        if temperature is None:
            temperature = tune_temperature

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    name = data_name + "_" + str(n_tasks) + "_tasks" + "/" + model_name + "/" + optimizer_type +  "/ssl/lr_" + str(learning_rate) + "_temp_" + str(temperature) + "_pre"
    print("Experiment name: " ,name)

    # HANDLE DEVICE
    device = get_device(device)

    # GET DATA
    pre_scenario = data.get_data(data_name, n_tasks=n_tasks, seed=seed) 
    class_scenario = data.get_data(data2_name, seed=seed)

    # CREATE MODEL
    num_classes = len([item for sublist in class_scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model("SimCLR_" + model_name, device, num_classes)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = ssl_get_eval_plugin(name="log/"+ name) 

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # SETUP OTHER PLUGINS
    # Construct the plugins
    if early_stopping > 0:
        plugins =[
            EarlyStoppingPlugin(early_stopping, "train_stream"),
            LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_epochs))
        ]
    else:
        plugins = [LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=ssl_epochs))]

    # DEFINE THE AUGMENTATIONS
    # we define lambda functions for the augmentations
    _, og_height, og_width = pre_scenario.original_train_dataset[0][0].shape
    random_resized_crop_lambda = transforms.Lambda(
        lambda imgs: 
            stack(
                [transforms.RandomResizedCrop(size=(og_height, og_width), scale=(0.2, 0.8), antialias=True)(img) for img in imgs]
            )
    )
    color_distort_lambda = transforms.Lambda(
        lambda imgs: 
            stack(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img) for img in imgs]
            )
    )
    # we create a composition of transformations including the lambda functions
    augs = transforms.Compose([
        random_resized_crop_lambda,    # Apply random cropping and resizing to original size
        color_distort_lambda           # Apply color distortion
    ]) 

    # PERFORM SSL
    # construct the strategy
    ssl_strategy = ss.SimCLR(
        model, optimizer,
        augmentations=augs, temperature=temperature,
        train_mb_size=ssl_batch_size, train_epochs=ssl_epochs, eval_mb_size=class_batch_size,
        evaluator=eval_plugin,
        device = device,
        plugins=plugins
    )
    # train
    pretrain(pre_scenario, ssl_strategy, name, device)

    # TRAIN CLASSIFIER
    name = name[:-4] + "_classification"
    eval_plugin = get_eval_plugin(name="log/"+ name, track_classes=[j for i in class_scenario.original_classes_in_exp for j in i])
    eval_strategy = ss.SimCLR(
        model, optimizer,
        augmentations=augs, temperature=temperature,
        train_mb_size=class_batch_size, train_epochs=class_epochs, eval_mb_size=class_batch_size,
        evaluator=eval_plugin,
        device = device
    )
    eval_strategy.done_pretraining()
    # train
    train(class_scenario, eval_strategy, name, device)  

# TODO: check ssl tuning is working correctly, check tensor logs etc.
# TODO: tidy the ssl, split into two different training routines and make it so SimCLR only does the ssl, not the classification (that should be done in a separate training routine)
# TODO: Buffer for ssl

# TODO: set the scipy RNG in set_seed so can remove the need for passing seed to get_data
# TODO: if you run experiment from checkpoint does logging continue from where it left off?