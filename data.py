from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitImageNet, SplitTinyImageNet, SplitMNIST
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.utils import make_classification_dataset
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

@dispatch(str)
def get_data(name, n_tasks=5, seed=None):
    """
    Function to get training and testing data
    downloads and stores data in folder named "data" if not already there
    Returns tuple consisting of testing and training data (in that order), unless is CL benchmark in which case returns just the scenario
    """
    if(name == "MNIST"):
        data = SplitMNIST(n_experiences=1, shuffle=False, return_task_id=False)
        return 
    elif(name == "CIFAR10"):
        data = SplitCIFAR10(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "CIFAR100"):
        data = SplitCIFAR100(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "ImageNet"):
        data = SplitImageNet(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "TinyImageNet"):
        data = SplitTinyImageNet(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "SplitCIFAR10"):
        data = SplitCIFAR10(n_experiences=n_tasks, shuffle=False, return_task_id=False)
        return data
    elif(name == "SplitCIFAR100"):
        data = SplitCIFAR100(n_experiences=n_tasks, shuffle=False, return_task_id=False)
        return data
    elif(name == "SplitImageNet"):
        data = SplitImageNet(n_experiences=n_tasks, shuffle=False, return_task_id=False)
        return data
    elif(name == "SplitTinyImageNet"):
        data = SplitTinyImageNet(n_experiences=n_tasks, shuffle=False, return_task_id=False)
        return data
    else:
        raise Exception("Not given valid dataset name must be: MNIST, CIFAR10, SplitCIFAR10, CIFAR100, SplitCIFAR100, SplitImageNet or SplitTinyImageNet")
   
@dispatch(str, str)
def get_data(name, name2, n_tasks=5, strategy={"name":"stratify", "percentage":0.5}, seed=None):
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
            test_data = data.original_test_dataset
            if(strategy["name"] == "stratify"):
                data1, data2 = stratify_split(data.original_train_dataset, strategy["percentage"], seed=seed)
                scenario = nc_benchmark(
                    data1,
                    test_data, 
                    n_experiences=n_tasks, 
                    shuffle=False, 
                    seed=seed, 
                    task_labels=False
                    )
                data2 = make_classification_dataset(data2, task_labels=-1)
                return (scenario, data2)
            else:
                raise Exception("Not given valid dataset split method currently only support: stratify")
        else:
            raise Exception("Not given valid dataset-2 name, currently only support partitioning SplitCIFAR10") 
    else:
        raise Exception("Not given valid dataset-1 name, currently only support: SplitCIFAR10")