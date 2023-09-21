from sklearn.model_selection import train_test_split
from avalanche.benchmarks.classic import SplitCIFAR10, SplitCIFAR100, SplitImageNet, SplitTinyImageNet, SplitMNIST
from avalanche.benchmarks.datasets import MNIST, CIFAR10, CIFAR100
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.utils import make_classification_dataset
from torchvision.transforms import ToTensor
from torch.utils.data import Subset
from multipledispatch import dispatch
from torchvision import transforms

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


cifar10_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

no_aug_cifar10_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

cifar100_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

no_aug_cifar100_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ]
)

@dispatch(str)
def get_data(name, n_tasks=5, seed=None, no_augmentation=False):
    """
    Function to get training and testing data
    downloads and stores data in folder named "data" if not already there
    Returns tuple consisting of testing and training data (in that order), unless is CL benchmark in which case returns just the scenario
    """
    if(name == "MNIST"):
        data = SplitMNIST(n_experiences=1, shuffle=False, return_task_id=False)
        return 
    elif(name == "CIFAR10"):
        if no_augmentation:
            data = SplitCIFAR10(n_experiences=1, shuffle=False, return_task_id=False, train_transform=no_aug_cifar10_transform)
        else: 
            data = SplitCIFAR10(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "CIFAR100"):
        if no_augmentation:
            data = SplitCIFAR100(n_experiences=1, shuffle=False, return_task_id=False, train_transform=no_aug_cifar100_transform)
        else:
            data = SplitCIFAR100(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "ImageNet"):
        data = SplitImageNet(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "TinyImageNet"):
        data = SplitTinyImageNet(n_experiences=1, shuffle=False, return_task_id=False)
        return data
    elif(name == "SplitCIFAR10"):
        if no_augmentation:
            data = SplitCIFAR10(n_experiences=n_tasks, shuffle=False, return_task_id=False, train_transform=no_aug_cifar10_transform)
        else:
            data = SplitCIFAR10(n_experiences=n_tasks, shuffle=False, return_task_id=False)
        return data
    elif(name == "SplitCIFAR100"):
        if no_augmentation:
            data = SplitCIFAR100(n_experiences=n_tasks, shuffle=False, return_task_id=False, train_transform=no_aug_cifar100_transform)
        else:
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
def get_data(name, name2, n_tasks=5, strategy={"name":"stratify", "percentage":0.5}, seed=None, no_augmentation=False):
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
        train_data = CIFAR10(root="./data", download=True, train=True)
        test_data = CIFAR10(root="./data", download=True, train=False)
        if(name2 == "SplitCIFAR10"):
            if(strategy["name"] == "stratify"):
                data1, data2 = stratify_split(train_data, strategy["percentage"], seed=seed)
                if no_augmentation:
                    train_aug = no_aug_cifar10_transform
                else:
                    train_aug = cifar10_transform
                scenario = nc_benchmark(
                    data1,
                    test_data, 
                    n_experiences=n_tasks, 
                    shuffle=False, 
                    seed=seed, 
                    task_labels=False,
                    train_transform = train_aug,
                    eval_transform = no_aug_cifar10_transform
                )
                data2 = make_classification_dataset(data2, task_labels=2, transform = train_aug)
                return (scenario, data2)
            else:
                raise Exception("Not given valid dataset split method currently only support: stratify")
        else:
            raise Exception("Not given valid dataset-2 name, currently only support partitioning SplitCIFAR10") 
    else:
        raise Exception("Not given valid dataset-1 name, currently only support: SplitCIFAR10")
    

# TODO: get rid of no_augmentation and set to whatever is the best for given dataset