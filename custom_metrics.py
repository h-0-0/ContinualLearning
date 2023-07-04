from avalanche.evaluation.metrics.accuracy import AccuracyPluginMetric


class ExpSpecificEpochAccuracy(AccuracyPluginMetric):
    """
    The average accuracy over a single training epoch.
    This plugin metric only works at training time.

    The accuracy will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super(ExpSpecificEpochAccuracy, self).__init__(
            reset_at="epoch", emit_at="epoch", mode="eval"
        )

    def __str__(self):
        return "Top1_Acc_Epoch"