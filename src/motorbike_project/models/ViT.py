import torch
import torch.nn as nn
import timm


class VisionTransformer(nn.Module):
    def __init__(self, num_classes: int = 3):
        super(VisionTransformer, self).__init__()

        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = VisionTransformer()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
