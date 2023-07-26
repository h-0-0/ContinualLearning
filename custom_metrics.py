from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name


class MyPluginMetric(PluginMetric[float]):
    """
    This metric will return a `float` value after
    each training epoch
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self._accuracy_metric = Accuracy()

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._accuracy_metric.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._accuracy_metric.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
            
        self._accuracy_metric.update(strategy.mb_output, strategy.mb_y, 
                                     task_labels)

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the epoch begins
        """
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        return self._package_result(strategy)
        
        
    def _package_result(self, strategy):
        """Taken from `GenericPluginMetric`, check that class out!"""
        metric_value = self.accuracy_metric.result()
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(self, strategy,
                                          add_experience=add_exp,
                                          add_task=True)
            return [MetricValue(self, metric_name, metric_value,
                                plot_x_position)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Top1_Acc_Epoch"