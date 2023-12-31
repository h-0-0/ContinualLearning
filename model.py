from torch import nn, load
# from torchvision.models import resnet18, resnet50
from torch.nn.functional import relu, avg_pool2d

# Class for VGG16 convolutional neural network architecture
class VGG16(nn.Module):
    def __init__(self, num_classes: int):

        super(VGG16, self).__init__()

        # Based on VGG16 architecture
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),

            # Block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),

            # Block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),

            # Block 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),

            # Block 5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self.flatten = nn.Flatten()
        self.device = None

    # Compute forward pass
    def forward(self, x):
        out = self.conv(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out
    
# # Class for ResNet18 from torchvision
# class ResNet18(nn.Module):
#     def __init__(self, num_classes: int):
#         super(ResNet18, self).__init__()
#         self.model = resnet18(weights=None, num_classes=num_classes)
#         # self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # self.model.fc = nn.Linear(512, num_classes)
#         self.device = None

#     # Compute forward pass
#     def forward(self, x):
#         return self.model(x)
    
# Class for ResNet50 from torchvision
# class ResNet50(nn.Module):
#     def __init__(self, num_classes: int):
#         super(ResNet50, self).__init__()
#         self.model = resnet50(weights=None, num_classes=num_classes)
#         # self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         # self.model.fc = nn.Linear(2048, num_classes)
#         self.device = None

#     # Compute forward pass
#     def forward(self, x):
#         return self.model(x)


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, config={}):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.conv2 = conv3x3(planes, planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
						  stride=stride, bias=False),
			)
		self.IC1 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

		self.IC2 = nn.Sequential(
			nn.BatchNorm2d(planes),
			nn.Dropout(p=config['dropout'])
			)

	def forward(self, x):
		out = self.conv1(x)
		out = relu(out)
		out = self.IC1(out)

		out += self.shortcut(x)
		out = relu(out)
		out = self.IC2(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes, nf, config={}):
		super(ResNet, self).__init__()
		self.in_planes = nf

		self.conv1 = conv3x3(3, nf * 1)
		self.bn1 = nn.BatchNorm2d(nf * 1)
		self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
		self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
		self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
		self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
		self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride, config):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride, config=config))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, task_id=None):
		bsz = x.size(0)
		out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		if task_id is not None:
			t = task_id
			offset1 = int((t-1) * 5)
			offset2 = int(t * 5)
			if offset1 > 0:
				out[:, :offset1].data.fill_(-10e10)
			if offset2 < 100:
				out[:, offset2:100].data.fill_(-10e10)
		return out


def resnet18(num_classes=10, nf=20, config={'dropout': 0.5}):
	net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, config=config)
	return net

def resnet50(num_classes=10, nf=20, config={'dropout': 0.5}):
	net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes, nf, config=config)
	return net

# Main function that instantiates all models and prints their architectures
# to facilitate a quick view of avaliable models
if __name__ == "__main__":
    print("VGG16 with 10 classes")
    print(VGG16(10))
    print("resnet18 with 10 classes")
    print(resnet18(10))
    print("resnet50 with 10 classes")
    print(resnet50(10))