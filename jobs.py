from slune.slune import lsargs, sbatchit
from slune.searchers import SearcherGrid

def reg_batch():
    search = SearcherGrid({
        "model_name": ["VGG16","resnet18", "resnet50"]
    })
    sbatchit(
        "main.py", 
        "gpu_template.sh", 
        search, 
        cargs=[
            "--epochs=300", 
            "--optimizer_type=SGD",
            "--n_tasks=1",
        ]
    )

def reg_cl():
    search = SearcherGrid({
        "model_name": ["VGG16","resnet18", "resnet50"],
        "epochs": [1, 5, 10, 50]
    })
    sbatchit(
        "main.py", 
        "gpu_template.sh", 
        search,
        cargs=[
            "--optimizer_type=SGD",
            "--n_tasks=5",
        ]
    )

def strat():
    search = SearcherGrid({
        "model_name": ["VGG16","resnet18", "resnet50"],
        "batch_ratio": {
            "percentage" : [0.2, 0.4, 0.6, 0.8]
        }
    })
    sbatchit(
        "main.py", 
        "gpu_template.sh", 
        search,
        cargs=[
            "--epochs=50", 
            "--optimizer_type=SGD",
            "--n_tasks=5",
            "--strategy=fixed_replay_stratify",
            "--data2_name=SplitCIFAR10",
        ]
    )

def ssl_batch():
    search = SearcherGrid({
        "model_name": ["VGG16","resnet18", "resnet50"]
    })
    sbatchit(
        "main.py", 
        "gpu_template.sh", 
        search,
        cargs=[
            "--ssl_epochs=100", 
            "--class_epochs=100",
            "--optimizer_type=SGD",
            "--n_tasks=1",
            "--strategy=ssl",
            "--data_name=SplitCIFAR10",
            "--data2_name=CIFAR10",
            "--ssl_batch_size=512",
        ]
    )

# TODO: change defaults in main based on above cargs, so that we don't have to specify them here

if __name__ == "__main__":
    python_path, args = lsargs()
    if args[0] == "reg_batch":
        reg_batch()
    elif args[0] == "reg":
        reg_cl()
    elif args[0] == "strat":
        strat()
    elif args[0] == "ssl_batch":
        ssl_batch()
    else:
        raise ValueError("Unknown experiment type: " + args[0])