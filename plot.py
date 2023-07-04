import matplotlib.pyplot as plt

def training_acc_plot(all_metrics: dict, level: str = "Epoch"):
    """Creates a plot with separate learning curves for each experience.

    :param all_metrics: Dictionary of metrics as returned by
        EvaluationPlugin.get_all_metrics
    :return: matplotlib figure
    """
    if level == "Epoch":
        accs_keys = list(filter(lambda x: "Top1_Acc_Epoch" in x, all_metrics.keys()))
    elif level == "Experience":
        accs_keys = list(filter(lambda x: "Top1_Acc_MB" in x, all_metrics.keys()))
    else:
        raise ValueError("level must be 'Epoch' or 'MB'")
    fig, ax = plt.subplots()
    for ak in accs_keys:
        k = ak.split("/")[-1]
        x, y = all_metrics[ak]
        plt.plot(x, y)
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(level + " Accuracy")
    return fig

def training_loss_plot(all_metrics: dict, level: str = "Epoch"):
    """Creates a plot with separate learning curves for each experience.

    :param all_metrics: Dictionary of metrics as returned by
        EvaluationPlugin.get_all_metrics
    :return: matplotlib figure
    """
    if level == "Epoch":
        accs_keys = list(filter(lambda x: "Top1_Acc_Epoch" in x, all_metrics.keys()))
    elif level == "Experience":
        accs_keys = list(filter(lambda x: "Top1_Acc_MB" in x, all_metrics.keys()))
    else:
        raise ValueError("level must be 'Epoch' or 'MB'")
    fig, ax = plt.subplots()
    print("accs_keys: ",accs_keys)
    for ak in accs_keys:
        k = ak.split("/")[-1]
        print("k: ",k)
        x, y = all_metrics[ak]
        print("x: ",x)
        print("y: ",y)
        plt.plot(x, y, label=k)
    ax.legend()
    ax.set_xlabel("Iterations")
    ax.set_ylabel(level + " Accuracy")
    return fig