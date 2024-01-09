import torch
import torch.nn as nn
from torchvision.models import resnet152


class ResNet152(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.resnet152 = resnet152(pretrained=True)  # Weights here are pretrained on ImageNet
        self.resnet152.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet152(x)


if __name__ == '__main__':
    model = ResNet152()
    print(model)
    x = torch.randn(1, 3, 120, 120)  # Params: (batch_size, channels, height, width)
    y = model(x)
    print(y.shape)
