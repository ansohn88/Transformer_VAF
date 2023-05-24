import pytorch_lightning as pl
import torch
from svg_model import AttnModel
from torch.nn import functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)


class EngineAttn(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        attn_dim: int,
        n_heads: int,
        depth: int,
    ) -> None:
        super().__init__()

        # init model
        self.model = AttnModel(
            input_dim=input_dim,
            out_dim=2,
            attn_dim=attn_dim,
            num_heads=n_heads,
            depth=depth,
        )

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Metrics
        average = "weighted"
        self.acc = MulticlassAccuracy(num_classes=2, average=average)
        self.recall = MulticlassRecall(num_classes=2, average=average)
        self.specificity = MulticlassSpecificity(num_classes=2, average=average)
        self.prec = MulticlassPrecision(num_classes=2, average=average)
        self.f1_score = MulticlassF1Score(num_classes=2, average=average)
        self.auroc = MulticlassAUROC(num_classes=2, average=average)
        self.auprc = MulticlassAveragePrecision(num_classes=2, average=average)

        self.confmat = ConfusionMatrix(task="binary", num_classes=2)

    def flatten_loss(self, preds, labels):
        # instance_loss = F.binary_cross_entropy_with_logits(
        #     preds, labels.float())
        if preds.ndim == 3:
            preds = preds.view((preds.size(1), 2))
            labels = labels.view((labels.size(1),))

        instance_loss = self.loss_fn(preds, labels)
        num_classes = len(labels.unique(return_counts=True)[0])

        if num_classes == 2:
            # [instances, 1, 2]
            weights = F.one_hot(
                labels.unsqueeze(1).long(), num_classes=num_classes
            ).float()
            # [instances, 1]
            weights = torch.sum(weights / weights.sum(dim=0, keepdim=True), dim=2)
            loss = torch.sum(instance_loss * weights, dim=0) / weights.sum(dim=0)
            return loss
        else:
            return instance_loss.unsqueeze_(0)

    def training_step(self, batch, batch_idx):
        x = batch["counts"]
        y = batch["labels"]

        svg_logits = self.model(x)

        loss = self.flatten_loss(preds=svg_logits, labels=y)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            #  prog_bar=True,
            logger=True,
            batch_size=1,
        )

        self.training_step_outputs.append(loss)

        return {
            "loss": loss,
        }

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean(dim=0)

        self.training_step_outputs.clear()

        self.log(
            "avg_train_loss",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch["counts"]
        y = batch["labels"]

        svg_logits = self.model(x)

        loss = self.flatten_loss(preds=svg_logits, labels=y)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=1
        )
        self.validation_step_outputs.append(loss)

        return {"loss": loss}

    @torch.no_grad()
    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean(dim=0)
        self.log(
            "avg_val_loss",
            avg_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        self.log("checkpoint_on", avg_val_loss)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x = batch["counts"]
        y = batch["labels"]

        svg_logits = self.model(x)

        loss = self.flatten_loss(preds=svg_logits, labels=y)

        y_probs = torch.nn.Softmax(dim=-1)(svg_logits)

        test_out = {
            "loss": loss,
            "svg_logits": svg_logits,
            "y_probs": y_probs,
            "targets": y,
        }
        self.test_step_outputs.append(test_out)

    @torch.no_grad()
    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()

        test_logits = []
        test_probs = []
        test_targets = []
        for x in self.test_step_outputs:
            svg_logit = x["svg_logits"].squeeze(0)
            probs = x["y_probs"].squeeze(0)
            target = x["targets"].squeeze(0)

            test_logits.append(svg_logit)
            test_probs.append(probs)
            test_targets.append(target)

        test_logits = torch.cat(test_logits, dim=0)
        test_probs = torch.cat(test_probs, dim=0)
        test_targets = torch.cat(test_targets, dim=0)

        test_preds = torch.topk(test_probs, 1, dim=1)[1]

        # print(
        #     f"""
        #     logits size: {test_logits.size()}
        #     probs size: {test_probs.size()}
        #     preds size: {test_preds.size()}
        #     targets size: {test_targets.size()}
        # """
        # )

        self.log("avg_test_loss", avg_loss, batch_size=1)

        # METRICS
        acc = self.acc(test_preds.long(), test_targets.long())
        spec = self.specificity(test_preds.long(), test_targets.long())
        recall = self.recall(test_preds.long(), test_targets.long())
        prec = self.prec(test_preds.long(), test_targets.long())
        f1 = self.f1_score(test_preds.long(), test_targets.long())
        auroc = self.auroc(test_probs.float(), test_targets.squeeze(1).long())
        auprc = self.auroc(test_probs.float(), test_targets.squeeze(1).long())
        cm = self.confmat(test_preds.long(), test_targets.long())

        print(
            f"""
            acc: {acc.item()}
            spec: {spec.item()}
            auroc: {auroc.item()}
            recall: {recall.item()}
            prec: {prec.item()}
            auprc: {auprc.item()}
        """
        )

        self.test_results = {
            "test_loss": avg_loss.cpu().numpy(),
            "test_svg_logits": test_logits.cpu().numpy(),
            "test_probs": test_probs.cpu().numpy(),
            "test_targets": test_targets.cpu().numpy(),
            "test_acc": acc.cpu().numpy(),
            "test_spec": spec.cpu().numpy(),
            "test_recall": recall.cpu().numpy(),
            "test_prec": prec.cpu().numpy(),
            "test_f1_score": f1.cpu().numpy(),
            "test_auroc": auroc.cpu().numpy(),
            "test_auprc": auprc.cpu().numpy(),
            "test_cm": cm.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=1e-5
        )

        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=2e-4,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.2,
            cycle_momentum=True,
        )

        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": schedule, "interval": "step"},
        }
