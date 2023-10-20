from slune.slune import sbatchit, get_csv_slog, garg, lsargs
from slune.searchers import SearcherGrid


from utils import get_device, get_model, get_optimizer, get_eval_plugin, ssl_get_eval_plugin, get_augmentations, done_train_ssl
import data
from avalanche.training.supervised.strategy_wrappers import Naive
from torch.nn import CrossEntropyLoss
import self_supervised as ss
from torch.optim import lr_scheduler
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin

def classification(params):
    """
    Runs the experiment with the given config in the batch scenario
    """
    print("Running classification tune with params: ", params)
    # SETUP SLOG
    slog = get_csv_slog(params)

    # GET PARAMS FROM CONFIG
    data_name = garg(params, "data_name")
    model_name = garg(params, "model_name")
    optimizer_type = garg(params, "optimizer_type")

    learning_rate = float(garg(params, "learning_rate"))

    name = slog.current_path

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
    slog.log(
        {
        "final_train_accuracy": all_metrics["Top1_Acc_Epoch/train_phase/train_stream/Task000"][1][-1],
        "final_test_accuracy": all_metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1][-1],
        }
    )
    slog.save_collated()

def ssl(params):
    """
    Runs the experiment with the given config in the batch scenario for ssl
    """
    print("Running ssl tune with params: ", params)
    # SETUP SLOG
    slog = get_csv_slog(params)

    # GET PARAMS FROM CONFIG
    data_name = garg(params, "data_name")
    model_name = garg(params, "model_name")
    optimizer_type = garg(params, "optimizer_type")

    learning_rate = float(garg(params, "learning_rate"))
    temperature = float(garg(params, "temperature"))

    name = slog.current_path

    # HANDLE DEVICE
    device = get_device()

    # GET DATA
    ssl_scenario = data.get_data(data_name, n_tasks=1, no_augmentation=True) # We turn off augmentations as we will be doing them as part of SSL
    class_scenario = data.get_data(data_name, n_tasks=1)

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
    slog.log(
        {
        "final_train_accuracy": all_metrics["Top1_Acc_Epoch/train_phase/train_stream/Task000"][1][-1],
        "final_test_accuracy": all_metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"][1][-1],
        }
    )
    slog.save_collated()

if __name__ == "__main__":
    python_path, args = lsargs()
    print("Arguments received to tune.py: ", args)
    if args[0] == "--tune_type=classification":
        # If no further args given we submit a grid search over learning rate to SLURM using slune package
        if len(args) == 1:
            search = SearcherGrid({
                "model_name": ["VGG16","ResNet18", "ResNet50"],
                "learning_rate": [0.5, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
            })
            sbatchit(
                "tune.py", 
                "gpu_template.sh", 
                search, 
                cargs=[
                    "--tune_type=classification", 
                    "--data_name=CIFAR10", 
                    "--optimizer_type=SGD"
                ]
            )
        else:
            # For debugging
            classification(args)
    elif args[0] == "--tune_type=ssl":
        # If no further args given we submit a grid search over learning rate to SLURM using slune package
        if len(args) == 1:
            search = SearcherGrid({
                "model_name": ["VGG16","ResNet18", "ResNet50"],
                "learning_rate": [0.1, 0.01, 0.001],
                "temperature": [0.01, 0.1, 0.5, 1, 5, 10, 50, 100]
            })
            sbatchit(
                "tune.py", 
                "gpu_template.sh", 
                search, 
                cargs=[
                    "--tune_type=ssl", 
                    "--data_name=CIFAR10",
                    "--optimizer_type=SGD"
                ]
            )
        else:
            # For debugging
            ssl(args)
    else:
        raise ValueError("Invalid tune type, please use 'classification' or 'ssl'")
    print("Finished tune with args: ", args)
