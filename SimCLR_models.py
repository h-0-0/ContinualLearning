from torch import nn
from torch.nn import Module
import model

class SimCLRModel(Module):
    """
    SimCLR Model.
    Uses a feature extractor and projection network during training.
    The input is passed through a feature extractor and then normalized
    before being fed to the projection network or classifier.
    During evaluation we freeze the feature extractor and only train the classifier.
    """

    def __init__(self, feature_extractor: Module, projection: Module, classifier: Module):
        """
        :param feature_extractor: a pytorch module that given the input
            examples extracts the hidden features
        :param projection: a pytorch module that takes as input the output
            of the feature extractor
        :param classifier: a pytorch module that takes as input the output
            of the feature extractor and produces the logits
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projector = projection
        self.classifier = classifier
        self.pretraining = True

    def forward(self, x):
        x = self.feature_extractor(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        if self.pretraining:
            self.feature_extractor.requires_grad_(True)
            return self.projector(x)
        elif self.training:
            self.feature_extractor.requires_grad_(False)
            return self.classifier(x)
        else:
            return self.classifier(x)
        
class VGG16(SimCLRModel):
    """
    VGG16 Model.
    Uses a feature extractor and projection network during training,
    and the feature extractor (frozen) and classifier during eval.
    The input is passed through a feature extractor and then normalized
    before being fed to the projection network or classifier.
    Uses a linear classifier.
    """

    def __init__(self, num_classes: int, input_channels: int = 3):
        """
        :param num_classes: number of classes in the dataset
        """
        feature_extractor = nn.Sequential(model.VGG16(input_channels = input_channels, num_classes = num_classes).conv, nn.Flatten())
        projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        classifier = nn.Linear(512, num_classes)
        super().__init__(feature_extractor, projection, classifier)

class ResNet18(SimCLRModel):
    """
    ResNet18 Model.
    Uses a feature extractor and projection network during training,
    and the feature extractor (frozen) and classifier during eval.
    The input is passed through a feature extractor and then normalized
    before being fed to the projection network or classifier.
    Uses the same classifier as the ResNet18 in model.py, which is a linear layer.
    """

    def __init__(self, num_classes: int, input_channels: int = 3, classifier_type = None):
        """
        :param num_classes: number of classes in the dataset
        """
        resnet18 = model.ResNet18(input_channels = input_channels, num_classes = num_classes)
        feature_extractor = nn.Sequential(
            nn.Sequential(*list(resnet18.children())[:-1]),
            nn.Flatten()
        )
        projection = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        classifier = resnet18.model.fc
        super().__init__(feature_extractor, projection, classifier)

class ResNet50(SimCLRModel):
    """
    ResNet50 Model.
    Uses a feature extractor and projection network during training,
    and the feature extractor (frozen) and classifier during eval.
    The input is passed through a feature extractor and then normalized
    before being fed to the projection network or classifier.
    Uses the same classifier as the ResNet50 in model.py, which is a linear layer.
    """

    def __init__(self, num_classes: int, input_channels: int = 3):
        """
        :param num_classes: number of classes in the dataset
        """
        resnet50 = model.ResNet50(input_channels = input_channels, num_classes = num_classes)
        feature_extractor = nn.Sequential(
            nn.Sequential(*list(resnet50.children())[:-1]),
            nn.Flatten()
        )
        projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        classifier = resnet50.model.fc
        super().__init__(feature_extractor, projection, classifier)