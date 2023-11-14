import run
from argparse import ArgumentParser
import sys

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
            exp_lr_decay_gamma = args.exp_lr_decay_gamma,
        )
    elif args.strategy == "fixed_replay_stratify":
        if args.data_name != "SplitCIFAR10":
            raise ValueError("Only SplitCIFAR10 is supported for regular strategy")
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
            data2_name = args.data2_name, 
            batch_ratio = args.batch_ratio, 
            percentage = args.percentage,
            exp_lr_decay_gamma = args.exp_lr_decay_gamma,
        )
    elif args.strategy == "ssl":
        if args.data_name != "SplitCIFAR10":
            raise ValueError("Only SplitCIFAR10 is supported for regular strategy")
        run.ssl(
            data_name = args.data_name, 
            data2_name = args.data2_name,
            model_name = args.model_name, 
            ssl_batch_size = args.ssl_batch_size,
            class_batch_size = args.class_batch_size,
            learning_rate = args.learning_rate, 
            ssl_epochs = args.ssl_epochs,
            class_epochs = args.class_epochs,
            n_tasks = args.n_tasks,
            device = args.device, 
            optimizer_type = args.optimizer_type, 
            seed = args.seed,
            temperature=args.temperature,
            replay=args.replay,
            exp_lr_decay_gamma = args.exp_lr_decay_gamma,
        )
    

if __name__ == "__main__":
    # Create argument parser
    parser = ArgumentParser()

    # Arguments related to experiment
    parser.add_argument("--data_name", type=str, help="Name of dataset to use, default: 'SplitCIFAR10'", default="SplitCIFAR10")
    parser.add_argument("--model_name", type=str, help="Name of model to use, default: 'VGG16'", default="VGG16")
    parser.add_argument("--batch_size", type=int, help="Batch size, default: 10", default=10)
    parser.add_argument("--learning_rate", type=float, help="Learning rate if set will not perform tuning and will use the given learning rate, default: None", default=None)
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs, default: 1", default=1)

    # Arguments related to CL
    parser.add_argument("--n_tasks", type=int, help="Number of tasks, default: 5", default=5)
    parser.add_argument("--exp_lr_decay_gamma", type=float, help="Exponential learning rate decay factor, default: 0.8, use 1.0 for no decay", default=0.8)

    # Arguments for if we are using a second dataset
    parser.add_argument("--data2_name", type=str, help="Name of second dataset to use, default: None", default=None)

    # Arguments to define strategy to use
    parser.add_argument("--strategy", type=str, help="Which strategy to use, default: 'regular'", default="regular")

    # Arguments for stratify strategy
    parser.add_argument("--percentage", type=float, help="Percentage of the dataset to be assigned to first dataset, default: '0.50'", default=0.50)
    parser.add_argument("--batch_ratio", type=float, help="Percentage of batch that should be composed of samples from the first dataset, default: 0.50", default=0.50)

    # Argument for overriding which device to use
    parser.add_argument("--device", type=str, help="Which device to use, default: None", default=None)

    # Argument for setting the optimizer
    parser.add_argument("--optimizer_type", type=str, help="Which optimizer to use, default: 'SGD'", default="SGD")

    # Argument for setting the seed
    parser.add_argument("--seed", type=int, help="Seed for random number generator, default: 2026", default=2026)

    # Argument for self supervised learning
    parser.add_argument("--temperature", type=float, help="Temperature for self supervised learning, default: 0.5", default=0.5)
    parser.add_argument("--ssl_batch_size", type=int, help="Batch size for self supervised learning, default: 512", default=512)
    parser.add_argument("--class_batch_size", type=int, help="Batch size for training classifier after self supervised learning, default: 256", default=256)
    parser.add_argument("--ssl_epochs", type=int, help="Number of epochs for self supervised learning, default: 100", default=100)
    parser.add_argument("--class_epochs", type=int, help="Number of epochs for training classifier after self supervised learning, default: 100", default=100)
    parser.add_argument("--replay", type=int, help="Do we want to use a replay buffer, if 0 will not use one, if bigger than zero will use buffer of that size, default: 0", default=0)

    # Parse arguments
    args = parser.parse_args()

    # Check arguments are correct
    if not (args.optimizer_type in ["Adam", "SGD", "SGD_momentum"]):
        raise ValueError("Only Adam, SGD and SGD with momentum optimizers are supported at the moment")
    if not (args.strategy in ["regular", "fixed_replay_stratify", "ssl"]):
        raise ValueError("Only regular, fixed replay stratify and ssl strategies are supported at the moment")
    if(args.strategy == "fixed_replay_stratify"):
        if any(filter(lambda x: "--temperature" in x, sys.argv)):
            raise ValueError("temperature is for when using the ssl strategy")
        elif any(filter(lambda x: "--ssl_batch_size" in x, sys.argv)):
            raise ValueError("ssl_batch_size is for when using the ssl strategy")
        elif any(filter(lambda x: "--class_batch_size" in x, sys.argv)):
            raise ValueError("class_batch_size is for when using the ssl strategy")
        elif any(filter(lambda x: "--ssl_epochs" in x, sys.argv)):
            raise ValueError("ssl_epochs is for when using the ssl strategy")
        elif any(filter(lambda x: "--class_epochs" in x, sys.argv)):
            raise ValueError("class_epochs is for when using the ssl strategy")
        elif any(filter(lambda x: "--replay" in x, sys.argv)):
            raise ValueError("replay is for when using the ssl strategy")
    if(args.strategy == "ssl"):
        if any(filter(lambda x: "--batch_size" in x, sys.argv)):
            raise ValueError("Cannot set batch size when using ssl strategy, please set ssl_batch_size and class_batch_size instead")
        elif any(filter(lambda x: "--batch_ratio" in x, sys.argv)):
            raise ValueError("Cannot set batch ratio when using ssl strategy this is only for fixed replay stratify strategy")
        elif any(filter(lambda x: "--percentage" in x, sys.argv)):
            raise ValueError("Cannot set percentage when using ssl strategy this is only for fixed replay stratify strategy")
        elif any(filter(lambda x: "--epochs" in x, sys.argv)):
            raise ValueError("Cannot set epochs when using ssl strategy, please set ssl_epochs and class_epochs instead")

    
    # Run experiment
    main(args)