import motorbike_project as mp
import torch.nn as nn
import torch
import pytorch_lightning as pl


class ModelClass(pl.LightningModule):
    def __init__(self, model: str = 'resnet18', num_classes: int = 3):
        super().__init__()

        if model == 'resnet50':
            self.model = mp.ResNet50(num_classes=num_classes)
        elif model == 'vit':
            self.model = mp.VisionTransformerBase(num_classes=num_classes)
        elif model == 'vit_tiny':
            self.model = mp.VisionTransformerTiny(num_classes=num_classes)
        elif model == 'swinv2_base':
            self.model = mp.SwinV2Base(num_classes=num_classes)
        elif model == 'mobilenetv3_large':
            self.model = mp.MobileNetV3Large(num_classes=num_classes)
        elif model == 'resnet18':
            self.model = mp.ResNet18(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model_ckpt = r'D:\Installer\checkpoint3090\checkpoints\resnet18\resnet18\resnet18.ckpt'

    # Create an instance of your model
    model = ModelClass(model='resnet18')

    # Load only the common keys between the model and the checkpoint
    checkpoint = torch.load(model_ckpt, map_location=torch.device('cpu'))
    model_state_dict = model.state_dict()
    common_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_state_dict}
    model.load_state_dict(common_state_dict)

    # Your inference code
    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print(y.shape)
