import motorbike_project as mp
import torch.nn as nn
import torch
import pytorch_lightning as pl
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.utils import class_weight


class MotorBikeModel(pl.LightningModule):
    def __init__(self, labels_csv_path: str, model: str = 'resnet152', num_classes: int = 3, lr: float = 1e-4):
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

        # TODO: Add more models here if you want

        self.train_loss = mp.RunningMean()
        self.val_loss = mp.RunningMean()

        self.train_acc = mp.RunningMean()
        self.val_acc = mp.RunningMean()

        self.train_f1 = mp.RunningMean()
        self.val_f1 = mp.RunningMean()

        self.loss = nn.CrossEntropyLoss(
            weight=self._create_class_weight(labels_csv_path=labels_csv_path)
        )
        self.lr = lr

    def _create_class_weight(self, labels_csv_path: str):
        """
            Create class weight for the loss function
        """

        df = pd.read_csv(labels_csv_path)
        df.loc[df['answer'] > 2, 'answer'] = 2
        class_weights = class_weight.compute_class_weight('balanced', classes=df['answer'].unique(), y=df['answer'])
        return torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, x):
        return self.model(x)

    def _cal_loss_and_acc(self, batch):
        """
            Calculate loss and accuracy for a batch
        """

        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        with torch.no_grad():  # No need to calculate gradient here
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            f1 = f1_score(y_true=y.cpu(), y_pred=y_hat.argmax(dim=1).cpu(), average='macro')

        return loss, acc, f1

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)

        self.train_loss.update(loss.item(), batch[0].shape[0])
        self.train_acc.update(acc.item(), batch[0].shape[0])
        # self.train_f1.update(f1, batch[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)

        self.val_loss.update(loss.item(), batch[0].shape[0])
        self.val_acc.update(acc.item(), batch[0].shape[0])
        # self.val_f1.update(f1, batch[0].shape[0])

        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss(), sync_dist=True)
        self.log("train_acc", self.train_acc(), sync_dist=True)
        # self.log("train_f1", self.train_f1(), sync_dist=True)

        self.train_loss.reset()
        self.train_acc.reset()
        # self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss(), sync_dist=True)
        self.log("val_acc", self.val_acc(), sync_dist=True)
        # self.log("val_f1", self.val_f1(), sync_dist=True)

        self.val_loss.reset()
        self.val_acc.reset()
        # self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
