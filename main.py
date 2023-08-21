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
            n_tasks = args.n_tasks,
            device = args.device, 
            optimizer_type = args.optimizer_type, 
            seed = args.seed,
            early_stopping = args.early_stopping
        )
    elif args.strategy == "fixed_replay_stratify":
        run.fixed_replay_stratify(
            data_name = args.data_name, 
            model_name = args.model_name, 
            batch_size = args.batch_size,
            learning_rate = args.learning_rate, 
            epochs = args.epochs,
            n_tasks = args.n_tasks,
            device = args.device, 
            optimizer_type = args.optimizer_type, 
            seed = args.seed,
            early_stopping = args.early_stopping,
            data2_name = args.data2_name, 
            batch_ratio = args.batch_ratio, 
            percentage = args.percentage
        )
    elif args.strategy == "ssl":
        run.ssl(
            data_name = args.data_name, 
            model_name = args.model_name, 
            batch_size = args.batch_size,
            learning_rate = args.learning_rate, 
            epochs = args.epochs,
            n_tasks = args.n_tasks,
            device = args.device, 
            optimizer_type = args.optimizer_type, 
            seed = args.seed,
            early_stopping = args.early_stopping,
            temperature=args.temperature
        )
    

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser()

    # Arguments related to experiment
    parser.add_argument("--data_name", type=str, help="Name of dataset to use, default: 'SplitCIFAR10'", default="SplitCIFAR10")
    parser.add_argument("--model_name", type=str, help="Name of model to use, default: 'VGG16'", default="VGG16")
    parser.add_argument("--batch_size", type=int, help="Batch size, default: 256", default=256)
    parser.add_argument("--learning_rate", type=float, help="Learning rate if set will not perform tuning and will use the given learning rate, default: None", default=None)
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs, default: 1", default=1)
    parser.add_argument("--early_stopping", type=int, help="Number of epochs to wait before stopping (ie. the patience), if set to 0 then will turn off early stopping, default: 0", default=0)

    # Arguments related to CL
    parser.add_argument("--n_tasks", type=int, help="Number of tasks, default: 5", default=5)

    # Arguments for if we are using a second dataset
    parser.add_argument("--data2_name", type=str, help="Name of second dataset to use, default: None", default=None)
    parser.add_argument("--batch_ratio", type=float, help="Percentage of batch that should be composed of samples from the first dataset, default: 0.50", default=0.50)

    # Arguments to define strategy to use
    parser.add_argument("--strategy", type=str, help="Which strategy to use, default: 'regular'", default="regular")

    # Arguments for stratify strategy
    parser.add_argument("--percentage", type=float, help="Percentage of the dataset to be assigned to first dataset, default: '0.50'", default=0.50)

    # Argument for overriding which device to use
    parser.add_argument("--device", type=str, help="Which device to use, default: None", default=None)

    # Argument for setting the optimizer
    parser.add_argument("--optimizer_type", type=str, help="Which optimizer to use, default: 'SGD'", default="SGD")

    # Argument for setting the seed
    parser.add_argument("--seed", type=int, help="Seed for random number generator, default: 2026", default=2026)

    # Argument for self supervised learning
    parser.add_argument("--temperature", type=float, help="Temperature for self supervised learning, default: 0.5", default=0.5)

    # Parse arguments
    args = parser.parse_args()

    # Check arguments are correct
    if not (args.optimizer_type in ["Adam", "SGD", "SGD_momentum"]):
        raise ValueError("Only Adam, SGD and SGD with momentum optimizers are supported at the moment")
    if not (args.strategy in ["regular", "fixed_replay_stratify", "ssl"]):
        raise ValueError("Only regular, fixed replay stratify and ssl strategies are supported at the moment")
    
    # Run experiment
    main(args)