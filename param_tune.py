from ray import tune, init, shutdown, air
from utils import get_device, get_model, get_optimizer, get_eval_plugin
import data
from avalanche.training.supervised.strategy_wrappers import Naive
from torch.nn import CrossEntropyLoss
import pandas as pd
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin
import os

def run_config(config):
    """
    Runs the experiment with the given config in the batch scenario
    """
    # GET PARAMS FROM CONFIG
    print("Running with config: ", config) #TODO: remove
    data_name = config["data_name"]
    model_name = config["model_name"]
    optimizer_type = config["optimizer_type"]

    learning_rate = config["learning_rate"]

    # HANDLE DEVICE
    device = get_device(False)

    # GET DATA
    scenario = data.get_data(data_name, n_tasks=1)

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # DEFINE THE EVALUATION PLUGIN and LOGGERS#
    eval_plugin = get_eval_plugin(name="tune_log/"+ data_name + "/" + model_name + "/" + optimizer_type + "/lr_" + str(learning_rate) ) 

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    # uses early stopping with patience 3 and max 500 epochs
    cl_strategy = Naive(
        model, optimizer,
        criterion = CrossEntropyLoss(), train_epochs=2,
        train_mb_size = 256, eval_mb_size = 256,
        device = device,
        evaluator = eval_plugin,
        plugins = [EarlyStoppingPlugin(3, "train_stream")]
    )
    # TODO: change epochs to 200 from 2

    # TRAINING LOOP
    for experience in scenario.train_stream:
        # we train the model on the current experience
        cl_strategy.train(experience)

        # we test the model on the full test stream
        cl_strategy.eval(scenario.test_stream)
    all_metrics = eval_plugin.get_all_metrics()
    print(list(filter(lambda x: "Top1_Acc_Epoch" in x, eval_plugin.get_all_metrics().keys()))) #TODO: remove
    print("Top1_Acc_Epoch/train_phase/train_stream/Task000: ", all_metrics["Top1_Acc_Epoch/train_phase/train_stream/Task000"][1]) #TODO: remove
    print("Top1_Acc_Stream/eval_phase/test_stream/Task000: ", all_metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1]) #TODO: remove
    return {
        "final_train_accuracy": all_metrics["Top1_Acc_Epoch/train_phase/train_stream/Task000"][1][-1],
        "final_test_accuracy": all_metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1][-1],
    }

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
    # Get names
    exp_name = "tune_results"
    storage_path = "tuning/" + data_name + "/" + model_name + "/" + optimizer_type
    experiment_path = f"{storage_path}/{exp_name}"
    # Check if tuning has already been done, if so load and return results
    try: 
        restored_tuner = tune.Tuner.restore(experiment_path, trainable=run_config)
        result_grid = restored_tuner.get_results()
        best_lr = result_grid.get_best_result(metric=selection_metric, mode="max")
        print("Best learning rate: ", best_lr["selection_metric"]) #TODO remove
        return best_lr
    except:
        pass
    init()
    static_params = {
        "data_name": data_name,
        "model_name": model_name,
        "optimizer_type": optimizer_type,
        "should_checkpoint": True
    }
    trial_space = {
        "learning_rate": tune.grid_search([0.001])
    }
    # TODO: add more params to tune: [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    trial_space = {**static_params, **trial_space}
    train_model = tune.with_resources(run_config, {"gpu": 1})
    tuner = tune.Tuner(
        train_model,
        param_space=trial_space,
        run_config=air.RunConfig(
            name=exp_name,
            stop={"training_iteration": 100},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="final_train_accuracy",
                num_to_keep=5,
            ),
            storage_path=storage_path,
        )
    )
    # TODO: check number of epochs, does it stop at 100? if so change stop to big num eg. 500
    result_grid = tuner.fit()
    shutdown()
    best_lr = result_grid.get_best_result(metric=selection_metric, mode="max")
    print("Best learning rate: ", best_lr) #TODO remove
    print("Best learning rate: ", best_lr["selection_metric"]) #TODO remove
    return best_lr

# TODO: implement ray checkpointing