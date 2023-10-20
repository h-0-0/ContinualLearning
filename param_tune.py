from ray import tune, init, shutdown, air
from ray.air import session
from utils import get_device, get_model, get_optimizer, get_eval_plugin, ssl_get_eval_plugin, get_augmentations, done_train_ssl
import data
from avalanche.training.supervised.strategy_wrappers import Naive
from torch.nn import CrossEntropyLoss
import pandas as pd
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
import os
import self_supervised as ss
from torch.optim import lr_scheduler
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from torch import stack
from torchvision import transforms

def run_config(config):
    """
    Runs the experiment with the given config in the batch scenario
    """
    # GET PARAMS FROM CONFIG
    data_name = config["data_name"]
    model_name = config["model_name"]
    optimizer_type = config["optimizer_type"]

    learning_rate = config["learning_rate"]

    name= data_name + "/" + model_name + "/" + optimizer_type + "/lr_" + str(learning_rate)

    # HANDLE DEVICE
    device = get_device()

    # GET DATA
    scenario = data.get_data(data_name, n_tasks=1)

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS#
    eval_plugin = get_eval_plugin(name) 

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(
        model, optimizer,
        criterion = CrossEntropyLoss(), train_epochs=200,
        train_mb_size = 256, eval_mb_size = 256,
        device = device,
        evaluator = eval_plugin,
        plugins = [LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=200))] 
    )

    # TRAINING LOOP
    for experience in scenario.train_stream:
        # we train the model on the current experience
        cl_strategy.train(experience)

        # we test the model on the full test stream
        cl_strategy.eval(scenario.test_stream)
    all_metrics = eval_plugin.get_all_metrics()
    session.report(
        {
        "final_train_accuracy": all_metrics["Top1_Acc_Epoch/train_phase/train_stream/Task000"][1][-1],
        "final_test_accuracy": all_metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1][-1],
        }
    )

def ssl_run_config(config):
    """
    Runs the experiment with the given config in the batch scenario for ssl
    """
    # GET PARAMS FROM CONFIG
    data_name = config["data_name"]
    model_name = config["model_name"]
    optimizer_type = config["optimizer_type"]

    learning_rate = config["learning_rate"]
    temperature = config["temperature"]

    name = "tune_log/"+ data_name + "/" + model_name + "/" + optimizer_type + "/lr_" + str(learning_rate) + "_temp_" + str(temperature) + "_ssl"

    # HANDLE DEVICE
    device = get_device()

    # GET DATA
    ssl_scenario = data.get_data(data_name, n_tasks=1, no_augmentation=True) # We turn off augmentations as we will be doing them as part of SSL
    class_scenario = data.get_data("CIFAR10", n_tasks=1)

    # CREATE MODEL
    num_classes = len([item for sublist in class_scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model("SimCLR_" + model_name, device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS
    eval_plugin = ssl_get_eval_plugin(name) 

    # DEFINE THE AUGMENTATIONS
    augs = get_augmentations(ssl_scenario)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    ssl_strategy = ss.SimCLR(
        model, optimizer,
        augmentations=augs, temperature=temperature,
        train_epochs=100,
        train_mb_size = 512, eval_mb_size = 512,
        device = device,
        evaluator = eval_plugin,
        plugins = [LRSchedulerPlugin(lr_scheduler.CosineAnnealingLR(optimizer, T_max=100))]
    )

    # TRAINING LOOP
    # Perform ssl
    for experience in ssl_scenario.train_stream:
        # we train the model on the current experience
        ssl_strategy.train(experience)

        # we test the model on the full test stream
        ssl_strategy.eval(ssl_scenario.test_stream)
    done_train_ssl(model, optimizer)

    # Perform classification to test ssl
    name = name[:-4] + "_classification"
    eval_plugin = get_eval_plugin(name, track_classes=[j for i in class_scenario.original_classes_in_exp for j in i])
    class_strategy = Naive(
        model, optimizer,
        criterion = CrossEntropyLoss(), train_epochs = 100,
        train_mb_size = 256, eval_mb_size = 256,
        evaluator = eval_plugin,
        device = device
    )
    for experience in class_scenario.train_stream:
        # we train the model on the current experience
        class_strategy.train(experience)

        # we test the model on the full test stream
        class_strategy.eval(class_scenario.test_stream)

    all_metrics = eval_plugin.get_all_metrics()
    session.report(
        {
        "final_train_accuracy": all_metrics["Top1_Acc_Epoch/train_phase/train_stream/Task000"][1][-1],
        "final_test_accuracy": all_metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1][-1],
        }
    )

def tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="final_train_accuracy"):
    """ 
    Function to tune hyperparameters.
    We run tuning on the batch scenario.
    Currently only learning rate is tuned.
    Checks if setup has already been tuned and loads results if so.

    - Parameters to give
        - data_name: name of the dataset to use
        - model_name: name of the model to use
        - optimizer_type: type of optimizer to use

    - Parameters with defualt values
        - selection_metric: metric to select the best config on, avaliable metrics are:
            - final_train_accuracy
            - final_test_accuracy
    """
    # Search space
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    # Get names
    exp_name = data_name + "_" + model_name + "_" + optimizer_type
    storage_path = "/user/work/jd18380/ContinualLearning/tuning"
    experiment_path = f"{storage_path}/{exp_name}"
    # Check if tuning has already been done, if so load and return results
    try: 
        restored_tuner = tune.Tuner.restore(experiment_path, trainable=run_config, resume_errored=True)
        result_grid = restored_tuner.get_results()
        tuned_lrs = []
        for i in range(len(result_grid)):
            try:
                tuned_lrs.append(result_grid[i].metrics["config"]["learning_rate"])
            except:
                pass
        if set(lrs) <=  set(tuned_lrs):
            best_result= result_grid.get_best_result(metric=selection_metric, mode="max")
            best_lr = best_result.metrics["config"]["learning_rate"]
            return best_lr
        else:
            lrs = [lr for lr in lrs if lr not in tuned_lrs]
    except:
        print("No tuner found, starting new tuning experiment")
    init(ignore_reinit_error=True, num_cpus=1)
    static_params = {
        "data_name": data_name,
        "model_name": model_name,
        "optimizer_type": optimizer_type,
        "should_checkpoint": True
    }
    trial_space = {
        "learning_rate": tune.grid_search(lrs)
    }
    trial_space = {**static_params, **trial_space}
    train_model = tune.with_resources(run_config, {"gpu": 1})
    tuner = tune.Tuner(
        train_model,
        param_space=trial_space,
        run_config=air.RunConfig(
            name=exp_name,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="final_train_accuracy",
                checkpoint_score_order="max",
                num_to_keep=1,
            ),
            storage_path=storage_path,
        )
    )
    result_grid = tuner.fit()
    shutdown()
    return tune_hyperparams(data_name, model_name, optimizer_type, selection_metric)

def ssl_tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="final_train_accuracy"):
    """ 
    Function to tune hyperparameters.
    We run tuning on the batch scenario.
    Currently only learning rate is tuned.
    Checks if setup has already been tuned and loads results if so.

    - Parameters to give
        - data_name: name of the dataset to use
        - model_name: name of the model to use
        - optimizer_type: type of optimizer to use

    - Parameters with defualt values
        - selection_metric: metric to select the best config on, avaliable metrics are:
            - final_train_accuracy
            - final_test_accuracy
    """
    # Search space
    lrs = [0.01, 0.05, 0.1, 0.5]
    temps = [0.05, 0.1, 0.5, 1]
    # Get names
    exp_name = "ssl_" + data_name + "_" + model_name + "_" + optimizer_type
    storage_path = "/user/work/jd18380/ContinualLearning/tuning"
    experiment_path = f"{storage_path}/{exp_name}"
    # Check if tuning has already been done, if so load and return results
    try: 
        restored_tuner = tune.Tuner.restore(experiment_path, trainable=ssl_run_config, resume_errored=True)
        result_grid = restored_tuner.get_results()
        tuned_lrs = []
        tuned_temps = []
        for i in range(len(result_grid)):
            try:
                tuned_lrs.append(result_grid[i].metrics["config"]["learning_rate"])
                tuned_temps.append(result_grid[i].metrics["config"]["temperature"])
            except:
                pass
        if (set(lrs) <=  set(tuned_lrs)) and (set(temps) <=  set(tuned_temps)):
            best_result= result_grid.get_best_result(metric=selection_metric, mode="max")
            best_lr = best_result.metrics["config"]["learning_rate"]
            best_temp = best_result.metrics["config"]["temperature"]
            return best_lr, best_temp
        else:
            lrs = [lr for lr in lrs if lr not in tuned_lrs]
            temps = [temp for temp in temps if temp not in tuned_temps]
    except:
        print("No tuner found, starting new tuning experiment")
    init(ignore_reinit_error=True, num_cpus=1)
    static_params = {
        "data_name": data_name,
        "model_name": model_name,
        "optimizer_type": optimizer_type,
        "should_checkpoint": True
    }
    trial_space = {
        "learning_rate": tune.grid_search(lrs),
        "temperature": tune.grid_search(temps)
    }
    trial_space = {**static_params, **trial_space}
    train_model = tune.with_resources(ssl_run_config, {"gpu": 1})
    tuner = tune.Tuner(
        train_model,
        param_space=trial_space,
        run_config=air.RunConfig(
            name=exp_name,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="final_train_accuracy",
                checkpoint_score_order="max",
                num_to_keep=1,
            ),
            storage_path=storage_path,
        )
    )
    result_grid = tuner.fit()
    shutdown()
    return ssl_tune_hyperparams(data_name, model_name, optimizer_type, selection_metric)