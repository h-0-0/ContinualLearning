from torch.nn import CrossEntropyLoss
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.supervised.strategy_wrappers import Naive
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from custom_plugins import BatchSplitReplay, FixedBuffer, TaskExpLrDecay
from give_model_task import GiveModelTask
import data as data
from utils import set_seed, get_eval_plugin, ssl_get_eval_plugin, get_optimizer, get_device, get_model, get_augmentations, format_arguments
from train_utils import train, tune_hyperparams, done_train_ssl
import self_supervised as ss
from slune import get_csv_slog
import os

def regular(**kwargs):
    # SET THE SEED 
    set_seed(kwargs["seed"])
    # TODO: might have to move this to after loading checkpoint to avoid errors 
    # TODO: try running without setting seed and seeing if you can load checkpoint, if it works then we know that setting the seed is causing problems

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if kwargs['learning_rate'] is None:
        learning_rate = tune_hyperparams('classification', kwargs['data_name'], kwargs['model_name'], kwargs['optimizer_type'], selection_metric='final_train_accuracy')
    else:
        learning_rate = kwargs['learning_rate']

    # HANDLE DEVICE
    device = get_device(kwargs['device'])

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    params = format_arguments(**kwargs)
    slog = get_csv_slog(params, root_dir='.')
    name, _ = os.path.split(slog.get_current_path())
    print("Experiment name: ", name)

    # GET DATA
    scenario = data.get_data(kwargs['data_name'], n_tasks=kwargs['n_tasks'], seed=kwargs['seed'])

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(kwargs['model_name'], device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(kwargs['optimizer_type'], model, learning_rate)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name, track_classes=[j for i in scenario.original_classes_in_exp for j in i])

    # DEFINE PLUGINS 
    plugins = [TaskExpLrDecay(gamma=kwargs['exp_lr_decay_gamma'])]

    # CREATE THE STRATEGY INSTANCE
    if kwargs['mask']:
        cl_strategy = GiveModelTask(
            model, 
            optimizer,
            criterion = CrossEntropyLoss(), 
            train_epochs=kwargs['epochs'],
            train_mb_size = kwargs['batch_size'], 
            eval_mb_size = kwargs['batch_size'],
            evaluator=eval_plugin,
            device = device,
            plugins=plugins
        )
    else:
        cl_strategy = Naive(
            model, 
            optimizer,
            criterion=CrossEntropyLoss(), 
            train_epochs=kwargs['epochs'],
            train_mb_size=kwargs['batch_size'], 
            eval_mb_size = kwargs['batch_size'],
            evaluator=eval_plugin,
            device= device,
            plugins=plugins
        )

    # TRAINING LOOP
    train(scenario, cl_strategy, name, device)

def fixed_replay_stratify(**kwargs):
    # SET THE SEED 
    set_seed(kwargs['seed'])

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if kwargs['learning_rate'] is None:
        learning_rate = tune_hyperparams('classification', kwargs['data_name'], kwargs['model_name'], kwargs['optimizer_type'], selection_metric="final_train_accuracy")
    else:
        learning_rate = kwargs['learning_rate']
        
    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    params = format_arguments(**kwargs)
    slog = get_csv_slog(params, root_dir='')
    name, _ = os.path.split(slog.get_current_path())
    print("Experiment name: ", name)

    # HANDLE DEVICE
    device = get_device(kwargs['device'])

    # GET DATA
    scenario, buffer_data = data.get_data(kwargs['data_name'], kwargs['data2_name'], n_tasks=kwargs['n_tasks'], strategy={"name":"stratify", "percentage":kwargs['percentage']}, seed=kwargs['seed']) 

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(kwargs['model_name'], device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(kwargs['optimizer_type'], model, learning_rate)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = get_eval_plugin(name, track_classes={0: [0,1,2,3,4,5,6,7,8,9], 2: [0,1,2,3,4,5,6,7,8,9]}) #TODO: make this automatic for dataset

    # SETUP OTHER PLUGINS
    # Construct batch sizes for benchmark and replay
    bs_bench = int(kwargs['batch_size']*kwargs['batch_ratio'])
    bs_replay = kwargs['batch_size'] - bs_bench
    # Construct the plugins
    plugins = [
        BatchSplitReplay(FixedBuffer, buffer_data, max_size=len(buffer_data), bs1=bs_bench, bs2=bs_replay),
        TaskExpLrDecay(gamma=kwargs['exp_lr_decay_gamma'])
    ]

    # CREATE THE STRATEGY INSTANCE
    # Construct the strategy
    cl_strategy = SupervisedTemplate(
        model, optimizer,
        CrossEntropyLoss(), train_mb_size=kwargs['batch_size'], train_epochs=kwargs['epochs'], eval_mb_size=kwargs['batch_size'],
        evaluator=eval_plugin,
        device = device,
        plugins=plugins
    )

    # TRAINING LOOP
    train(scenario, cl_strategy, name, device)

def ssl(**kwargs):
    # SET THE SEED 
    set_seed(kwargs['seed'])

    # PERFORM/LOAD HYPERPARAMETER TUNING
    if kwargs['learning_rate'] is None or kwargs['temperature'] is None:
        tune_learning_rate, tune_temperature = tune_hyperparams('ssl', kwargs['data_name'], kwargs['model_name'], kwargs['optimizer_type'], selection_metric="final_train_accuracy")
        if kwargs['learning_rate'] is None:
            learning_rate = tune_learning_rate
        else:
            learning_rate = kwargs['learning_rate']
        if kwargs['temperature'] is None:
            temperature = tune_temperature
        else:
            temperature = kwargs['temperature']

    # CREATE NAME FOR LOGGING, CHECKPOINTING, ETC
    params = format_arguments(**kwargs)
    slog = get_csv_slog(params, root_dir='')
    name, _ = os.path.split(slog.get_current_path())
    print("Experiment name: ", name)

    # HANDLE DEVICE
    device = get_device(kwargs['device'])

    # GET DATA
    ssl_scenario = data.get_data(kwargs['data_name'], n_tasks=kwargs['n_tasks'], seed=kwargs['seed'], no_augmentation=True) # We turn off augmentations as we will be doing them as part of SSL
    class_scenario = data.get_data(kwargs['data2_name'], seed=kwargs['seed'])

    # CREATE MODEL
    num_classes = len([item for sublist in class_scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model("SimCLR_" + kwargs['model_name'], device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(kwargs['optimizer_type'], model, learning_rate)

    # SETUP OTHER PLUGINS
    # Construct the plugins
    plugins = [TaskExpLrDecay(gamma=kwargs['exp_lr_decay_gamma'])]
    if kwargs['replay'] > 0:
        storage_policy = ReservoirSamplingBuffer(max_size=kwargs['replay'])
        replay_plugin = ReplayPlugin(
            mem_size=kwargs['replay'], storage_policy=storage_policy
        )
        plugins.append(replay_plugin)

    # DEFINE THE AUGMENTATIONS
    augs = get_augmentations(ssl_scenario)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = ssl_get_eval_plugin(name) 

    # PERFORM SSL
    # construct the strategy
    ssl_strategy = ss.SimCLR(
        model, 
        optimizer,
        augmentations = augs, 
        temperature = temperature,
        train_mb_size = kwargs['ssl_batch_size'], 
        train_epochs = kwargs['ssl_epochs'], 
        eval_mb_size = kwargs['ssl_batch_size'],
        evaluator = eval_plugin,
        device = device,
        plugins = plugins
    )
    # train
    train(ssl_scenario, ssl_strategy, name, device)
    done_train_ssl(model, optimizer) # absorb this into training strategy, is there after all experiences callback?

    # TRAIN CLASSIFIER
    name = os.path.join(name, "classification")
    eval_plugin = get_eval_plugin(name, track_classes=[j for i in class_scenario.original_classes_in_exp for j in i])
    class_strategy = Naive(
        model, 
        optimizer,
        criterion = CrossEntropyLoss(), 
        train_epochs = kwargs['class_epochs'],
        train_mb_size = kwargs['class_batch_size'], 
        eval_mb_size = kwargs['class_batch_size'],
        evaluator = eval_plugin,
        device = device,
        # plugins = []
    )
    # train
    train(class_scenario, class_strategy, name, device)  

# TODO: set the scipy RNG in set_seed so can remove the need for passing seed to get_data
# TODO: if you run experiment from checkpoint does logging continue from where it left off?
# TODO: use slune to handle naming and saving of experiments