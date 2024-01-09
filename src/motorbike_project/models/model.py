import motorbike_project as mp
import torch.nn as nn
import torch
import pytorch_lightning as pl

from sklearn.metrics import f1_score, accuracy_score


class MotorBikeModel(pl.LightningModule):
    def __init__(self, model: str = 'resnet152', num_classes: int = 3, lr: float = 2e-4):
        super().__init__()

        if model == 'resnet152':
            self.model = mp.ResNet152(num_classes=num_classes)
        elif model == 'vit':
            self.model = mp.VisionTransformer(num_classes=num_classes)

        # TODO: Add more models here if you want

        self.train_loss = mp.RunningMean()
        self.val_loss = mp.RunningMean()

        self.train_acc = mp.RunningMean()
        self.val_acc = mp.RunningMean()

        self.train_f1 = mp.RunningMean()
        self.val_f1 = mp.RunningMean()

        self.loss = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _cal_loss_and_acc(self, batch):
        """
            Calculate loss and accuracy for a batch
        """

        x, y = batch
        y_hat = self(x)

        # Convert the y_hat from probability to class
        y_hat_float = torch.argmax(y_hat, dim=1).float()
        y_float = y.float()

        loss = self.loss(y_hat_float, y_float)
        loss.requires_grad = True

        with torch.no_grad():  # No need to calculate gradient here
            acc = (y_hat.argmax(dim=1) == y).float().mean()
            f1 = f1_score(y_true=y.cpu(), y_pred=y_hat.argmax(dim=1).cpu(), average='macro')

        return loss, acc, f1

    def training_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)

        self.train_loss.update(loss.item(), batch[0].shape[0])
        self.train_acc.update(acc.item(), batch[0].shape[0])
        self.train_f1.update(f1, batch[0].shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)

        self.val_loss.update(loss.item(), batch[0].shape[0])
        self.val_acc.update(acc.item(), batch[0].shape[0])
        self.val_f1.update(f1, batch[0].shape[0])

        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss(), sync_dist=True)
        self.log("train_acc", self.train_acc(), sync_dist=True)
        self.log("train_f1", self.train_f1(), sync_dist=True)

        self.train_loss.reset()
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss(), sync_dist=True)
        self.log("val_acc", self.val_acc(), sync_dist=True)
        self.log("val_f1", self.val_f1(), sync_dist=True)

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss, acc, f1 = self._cal_loss_and_acc(batch)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
