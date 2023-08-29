from torch import nn
from torch.nn import Module
import model

class SimCLRModel(Module):
    """
    SimCLR Model.
    Uses a feature extractor and projection network during training.
    The input is passed through a feature extractor and then normalized
    before being fed to the projection network or classifier.
    In ssl training mode (is_train_ssl = True), the model returns the output of the projection network.
    In classifier training mode (is_train_classifier = True), the model returns the output of the classifier.
    We initialize the model so that it is in the training SSL mode (ie. ready to be trained in a self supervised manner).
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
        self.is_train_ssl = True
        self.is_train_classifier = False

    def forward(self, x):
        x = self.feature_extractor(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        if self.is_train_ssl:
            self.feature_extractor.requires_grad_(True)
            return self.projector(x)
        elif self.is_train_classifier:
            self.feature_extractor.requires_grad_(False)
            return self.classifier(x)
        else:
            raise ValueError("Model must be in is_train_ssl or is_train_classifier mode.")
    
    def set_train_ssl(self):
        self.is_train_ssl = True
        self.is_train_classifier = False

    def set_train_classifier(self):
        self.is_train_ssl = False
        self.is_train_classifier = True

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