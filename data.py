from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.benchmarks.generators import nc_benchmark
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from multipledispatch import dispatch


def stratify_split(dataset, percentage, seed=None):
    """
    Function that takes in a dataset and splits it in stratified fashion into two datasets (with equal number of samples for each class) according to percentage given
    """
    data1_indices, data2_indices = train_test_split(
        range(len(dataset)),
        stratify=dataset.targets,
        train_size=percentage,
        test_size=1-percentage,
        random_state=seed
    )
    data1_split = Subset(dataset, data1_indices)
    data2_split = Subset(dataset, data2_indices)
    return data1_split, data2_split
# TODO: check this is splitting correctly

@dispatch(str, int)
def get_data(name, n_tasks=5):
    """
    Function to get training and testing data
    downloads and stores data in folder named "data" if not already there
    Returns tuple consisting of testing and training data (in that order), unless is CL benchmark in which case returns just the scenario
    """
    if(name == "MNIST"):
        training_data = MNIST(
            "data", 
            download=True, 
            train=True,
            transform=ToTensor()
        ),

        test_data = MNIST(
            "data", 
            download=True, 
            train=False,
            transform=ToTensor()
        )
        data = (training_data[0], test_data)
        return data
    elif(name == "CIFAR100"):
        training_data = CIFAR100(
            "data", 
            download=True, 
            train=True,
            transform=ToTensor()
        ),

        test_data = CIFAR100(
            "data", 
            download=True, 
            train=False,
            transform=ToTensor()
        )
        data = (training_data[0], test_data)
        return data
    elif(name == "CIFAR10"):
        training_data = CIFAR10(
            "data", 
            download=True, 
            train=True,
            transform=ToTensor()
        ),

        test_data = CIFAR10(
            "data", 
            download=True, 
            train=False,
            transform=ToTensor()
        )
        data = (training_data[0], test_data)
        return data
    elif(name == "SplitCIFAR10"):
        data = SplitCIFAR10(n_experiences=n_tasks, shuffle=False)
        return data
    else:
        raise Exception("Not given valid dataset name must be: MNIST, CIFAR10, SplitCIFAR10 or CIFAR100")
   
@dispatch(str, str, int, dict)
def get_data(name, name2, n_tasks=5, strategy={"name":"stratify", "percentage":0.5}):
    """
    Function to get training and testing data
    downloads and stores data in folder named "data" if not already there
    Returns tuple of two datasets according to dataset names and strategy given
    Available strategy's:
        - stratify: splits dataset into two datasets with equal number of samples for each class, percentage is the percentage of the original dataset to be assigned to the first dataset (the other percentage is assigned to the second dataset), currently first dataset will be a CL scenario and second dataset will be a normal dataset
            - name : "stratify"
            - percentage : float between 0 and 1
    """
    if(name == "SplitCIFAR10"):
        data = get_data("CIFAR10")
        if(name2 == "SplitCIFAR10"):
            test_data = data[1]
            if(strategy["name"] == "stratify"):
                # If Stratify is in the name we use the balanced split with number being the % to be assigned to task 1, ie. each dataset has the same number of samples per class    
                data1, data2 = stratify_split(data[0], strategy["percentage"])
                scenario = nc_benchmark(
                    data1,
                    test_data, 
                    n_experiences=n_tasks, 
                    shuffle=False, 
                    seed=None, 
                    task_labels=True
                    )
                return (scenario, data2)
            else:
                raise Exception("Not given valid dataset split method currently only support: stratify")
        else:
            raise Exception("Not given valid dataset-2 name, currently only support partitioning SplitCIFAR10") 
    else:
        raise Exception("Not given valid dataset-1 name, currently only support: SplitCIFAR10")