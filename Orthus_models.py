from torch.nn import Module, Sequential, Linear, ReLU, Flatten
import model
import copy

class OrthusModel(Module):
    """
    Orthus Model.
    Two headed model, one head is a projection network, the other is a classifier.
    We train the projection network on all data using a loss of your choice.
    We train the classifier on data from the buffer using a cross entropy loss, we assume data in the buffer iid.
    Make sure to train using the Orthus strategy so that we optimize both heads, 
    if you do not use the Orthus strategy make sure to deal with the two outputs of the forwards pass.
    """

    def __init__(self, feature_extractor: Module, projector: Module, classifier: Module):
        """
        :param feature_extractor: a pytorch module used to extract features
        :param projection: a pytorch module that takes as input the output
            of the feature extractor and produces an output to be optimized by a loss function, we train it on all data
        :param classifier: a pytorch module that takes as input the output
            of the feature extractor and tries to classify the input, we only train it on data from the buffer
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projector = projector
        self.classifier = classifier
        self.classifier_ind = None

    def forward(self, x):
        x = self.feature_extractor(x)
        # We create tensor to train classifier on, using the indices provided in classifier_ind
        x_buffer = x[self.classifier_ind]
        xp = self.projector(x)
        xc = self.classifier(x_buffer)
        return xp, xc
    
# -----------------------------
# The following are instantiations of the Orthus model with different feature extractors
# -----------------------------
class VGG16(OrthusModel):
    """
    VGG16 based Orthus network:
    Feature extractor with two heads (projection and classifier).
    """
    def __init__(self, num_classes: int, input_channels: int = 3):
        """
        :param num_classes: number of classes in the dataset
        """
        full_model = model.VGG16(input_channels = input_channels, num_classes = num_classes)
        feature_extractor = Sequential(full_model.conv, Flatten())
        projection = copy.deepcopy(full_model.fc)
        classifier = copy.deepcopy(full_model.fc)
        super().__init__(feature_extractor, projection, classifier)

class ResNet18(OrthusModel):
    """
    ResNet18 based Orthus network:
    Feature extractor with two heads (projection and classifier).
    """
    def __init__(self, num_classes: int, input_channels: int = 3):
        """
        :param num_classes: number of classes in the dataset
        """
        full_model = model.ResNet18(input_channels = input_channels, num_classes = num_classes)
        feature_extractor = Sequential(Sequential(*list(full_model.children())[:-1]), Flatten())
        projection = Linear(3072, num_classes)
        classifier = Linear(3072, num_classes)
        super().__init__(feature_extractor, projection, classifier)

class ResNet50(OrthusModel):
    """
    ResNet50 based Orthus network:
    Feature extractor with two heads (projection and classifier).
    """
    def __init__(self, num_classes: int, input_channels: int = 3):
        """
        :param num_classes: number of classes in the dataset
        """
        full_model = model.ResNet50(input_channels = input_channels, num_classes = num_classes)
        feature_extractor = Sequential(Sequential(*list(full_model.children())[:-1]), Flatten())
        projection = Linear(3072, num_classes)
        classifier = Linear(3072, num_classes)
        super().__init__(feature_extractor, projection, classifier)



# Main function that instantiates all models and prints their architectures
# to facilitate a quick view of avaliable models
if __name__ == "__main__":
    print("VGG16 with 10 classes")
    print(VGG16(10))
    print("ResNet18 with 10 classes")
    print(ResNet18(10))
    print("ResNet50 with 10 classes")
    print(ResNet50(10))