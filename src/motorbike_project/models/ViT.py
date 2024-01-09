import torch.nn as nn
from torchvision.models import vision_transformer


class VisionTransformer(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=3, lr=1e-4):
        super(VisionTransformer, self).__init__()
        self.model = vision_transformer.__dict__[model_name](pretrained=True)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.model.lr = lr

    def forward(self, x):
        return self.model(x)
