import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics


class FocalJaccardLoss(nn.Module):
    def __init__(self, num_classes, mode="multiclass", normalized=True):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(mode=mode, normalized=normalized)
        self.jaccard_loss = smp.losses.JaccardLoss(mode=mode, classes=num_classes)

    def forward(self, preds, targets):
        return self.focal_loss(preds, targets) + self.jaccard_loss(preds, targets)


class XEntJaccardLoss(nn.Module):
    def __init__(self, num_classes, mode="multiclass"):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.jaccard_loss = smp.losses.JaccardLoss(mode=mode, classes=num_classes)

    def forward(self, preds, targets):
        return self.ce_loss(preds, targets) + self.jaccard_loss(preds, targets)


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        model="Unet",
        backbone="resnet18",
        loss="ce_jaccard",
        num_channels=3,
        num_classes=2,
        weights=None,
        learning_rate=1e-3,
        optimizer="SGD",
    ):
        super().__init__()
        self.save_hyperparameters()
        model = getattr(smp, model)(
            encoder_name=backbone,
            encoder_weights=None if weights != "imagenet" else "imagenet",
            in_channels=num_channels,
            classes=num_classes,
        )
        self.model = model
        if loss == "ce_jaccard":
            self.loss_fn = XEntJaccardLoss(num_classes=num_classes, mode="multiclass")
        elif loss == "focal_jaccard":
            self.loss_fn = FocalJaccardLoss(
                num_classes=num_classes, mode="multiclass", normalized=True
            )
        elif loss == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss == "jaccard":
            self.loss_fn = smp.losses.JaccardLoss(
                mode="multiclass", classes=num_classes
            )
        else:
            raise ValueError("Unknown loss function")

        self.train_metrics = torchmetrics.MetricCollection(
            {
                "OverallAccuracy": torchmetrics.Accuracy(
                    num_classes=num_classes, average="micro", mdmc_average="global"
                ),
                "OverallPrecision": torchmetrics.Precision(
                    num_classes=num_classes, average="micro", mdmc_average="global"
                ),
                "OverallRecall": torchmetrics.Recall(
                    num_classes=num_classes, average="micro", mdmc_average="global"
                ),
                "OverallF1Score": torchmetrics.FBetaScore(
                    num_classes=num_classes,
                    beta=1.0,
                    average="micro",
                    mdmc_average="global",
                ),
                "AverageAccuracy": torchmetrics.Accuracy(
                    num_classes=num_classes, average="macro", mdmc_average="global"
                ),
                "AveragePrecision": torchmetrics.Precision(
                    num_classes=num_classes, average="macro", mdmc_average="global"
                ),
                "AverageRecall": torchmetrics.Recall(
                    num_classes=num_classes, average="macro", mdmc_average="global"
                ),
                "AverageF1Score": torchmetrics.FBetaScore(
                    num_classes=num_classes,
                    beta=1.0,
                    average="macro",
                    mdmc_average="global",
                ),
                "IoU": torchmetrics.JaccardIndex(
                    num_classes=num_classes, ignore_index=0
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def configure_optimizers(self):
        return getattr(torch.optim, self.hparams.optimizer)(
            self.model.parameters(), lr=self.hparams.learning_rate
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        self.train_metrics(y_hat_hard, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss_fn(y_hat, y)
        self.val_metrics(y_hat_hard, y)
        self.log("val_loss", loss, on_step=True, on_epoch=False)

        if batch_idx < 5:
            image = self.trainer.datamodule.plot(y_hat_hard[0])
            self.logger.experiment.add_image(
                "predictions/val", image, global_step=self.global_step + batch_idx
            )
            if self.current_epoch == 0:
                image = self.trainer.datamodule.plot(y[0])
                self.logger.experiment.add_image(
                    "ground-truth/val", image, global_step=self.global_step + batch_idx
                )

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()
