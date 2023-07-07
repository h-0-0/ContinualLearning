from ray import tune
from utils import get_device, get_model, get_optimizer
import data
from avalanche.training.supervised.strategy_wrappers import Naive
from torch.nn import CrossEntropyLoss
import pandas as pd
from avalanche.training.plugins.early_stopping import EarlyStoppingPlugin

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
    device = get_device(device)

    # GET DATA
    scenario = data.get_data(data_name, n_tasks=1)

    # CREATE MODEL
    num_classes = len([item for sublist in scenario.original_classes_in_exp for item in sublist]) # so we set the output layer to the correct size
    model = get_model(model_name, device, num_classes)

    # CREATE OPTIMIZER
    optimizer = get_optimizer(optimizer_type, model, learning_rate)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    # uses early stopping with patience 3 and max 500 epochs
    cl_strategy = Naive(
        model, optimizer,
        criterion = CrossEntropyLoss(), train_epochs=500,
        train_mb_size = 128, eval_mb_size = 128,
        device = device,
        plugins = [EarlyStoppingPlugin(3, "train_stream")]
    )

    # TRAINING LOOP
    train_results = []
    test_results = []
    for experience in scenario.train_stream:
        # train returns a dictionary which contains all the metric values
        train_results.append(cl_strategy.train(experience))

        # test also returns a dictionary which contains all the metric values
        test_results.append(cl_strategy.eval(scenario.test_stream))
    print("Completed a run with config: ", config)
    return {
        "final_train_accuracy": test_results[-1]["Top1_Acc_Epoch"], 
        "final_test_accuracy": test_results[-1]["Top1_Acc_Epoch"],
        "top_train_accuracy": max([result["Top1_Acc_Epoch"] for result in train_results]),
        "top_test_accuracy": max([result["Top1_Acc_Epoch"] for result in test_results])
    }

def tune_hyperparams(data_name, model_name, optimizer_type, selection_metric="top_test_accuracy"):
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
            - top_train_accuracy
            - top_test_accuracy
    """
    save_name = "tuning/" + data_name + "/" + model_name + "/" + optimizer_type +".csv"
    # Check if tuning has already been done, if so load and return results
    try: 
        df = pd.read_csv(save_name)
        # Get the best config
        best = df.loc[df[selection_metric].idxmax()]
        return best
    except:
        pass

    static_params = {
        "data_name": data_name,
        "model_name": model_name,
        "optimizer_type": optimizer_type,
    }
    trial_space = {
        "learning_rate": tune.grid_search([0.001, 0.01, 0.1])
    }
    trial_space = {**static_params, **trial_space}
    train_model = tune.with_resources(run_config, {"gpu": 1})
    tuner = tune.Tuner(
        train_model,
        param_space=trial_space
    )
    results = tuner.fit()
    # Get dataframe for analysis and save it to csv
    df = results.get_dataframe()
    df.to_csv(save_name)
    # Get the best config
    best = df.loc[df[selection_metric].idxmax()]
    return best

