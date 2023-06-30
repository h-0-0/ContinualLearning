import run
from argparse import ArgumentParser

# Main function
def main(args):
    if args.strategy == "regular":
        if args.data_name != "SplitCIFAR10":
            raise ValueError("Only SplitCIFAR10 is supported for regular strategy")
        run.regular(
            data_name = args.data_name, 
            model_name = args.model_name, 
            batch_size = args.batch_size,
            learning_rate = args.learning_rate, 
            epochs = args.epochs,
            load_model = args.load_model, 
            save_model = args.save_model,
            n_tasks = args.n_tasks,
            device = args.device, 
            optimizer_type = args.optimizer_type, 
            seed = args.seed
        )
    elif args.strategy == "fixed_replay_stratify":
        run.fixed_replay_stratify(
            data_name = args.data_name, 
            model_name = args.model_name, 
            batch_size = args.batch_size,
            learning_rate = args.learning_rate, 
            epochs = args.epochs,
            load_model = args.load_model, 
            save_model = args.save_model,
            n_tasks = args.n_tasks,
            device = args.device, 
            optimizer_type = args.optimizer_type, 
            seed = args.seed,
            data2_name = args.data2_name, 
            batch_ratio = args.batch_ratio, 
            percentage = args.percentage
        )
    

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser()

    # Arguments related to experiment
    parser.add_argument("--data_name", type=str, help="Name of dataset to use", default="SplitCIFAR10")
    parser.add_argument("--model_name", type=str, help="Name of model to use", default="VGG16")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=1)

    # Argument for loading model
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--no-load_model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=True)

    # Argument for saving model
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--no-save_model', dest='save_model', action='store_false')
    parser.set_defaults(save_model=True)

    # Arguments related to CL
    parser.add_argument("--n_tasks", type=int, help="Number of tasks", default=5)

    # Arguments for if we are using a second dataset
    parser.add_argument("--data2_name", type=str, help="Name of second dataset to use", default=False)
    parser.add_argument("--batch_ratio", type=float, help="Ratio of batch size to use for first dataset", default=0.50)

    # Arguments to define strategy to use
    parser.add_argument("--strategy", type=str, help="Which strategy to use", default="regular")

    # Arguments for stratify strategy
    parser.add_argument("--percentage", type=float, help="Percentage of original dataset to be assigned to first dataset", default=0.5)

    # Argument for overriding which device to use
    parser.add_argument("--device", type=str, help="Which device to use", default=False)

    # Argument for setting the optimizer
    parser.add_argument("--optimizer_type", type=str, help="Which optimizer to use", default="SGD")

    # Argument for setting the seed
    parser.add_argument("--seed", type=int, help="Seed for random number generator", default=2026)

    # Parse arguments
    args = parser.parse_args()

    # Check arguments are correct
    if not (args.optimizer_type in ["Adam", "SGD", "SGD_momentum"]):
        raise ValueError("Only Adam, SGD and SGD with momentum optimizers are supported at the moment")
    if not (args.strategy in ["regular", "fixed_replay_stratify"]):
        raise ValueError("Only regular and fixed replay stratify strategies are supported at the moment")
    
    # Run experiment
    main(args)