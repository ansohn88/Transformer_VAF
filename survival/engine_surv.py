import pickle

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

# import torch.nn as nn
from loss_surv import SurvLoss, loss_reg_l1
from model_surv import AttnModel, ConvAttnModel
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from utils_surv import create_struct_nparr, get_events_time


class EngineAttn(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        attn_dim: int,
        n_heads: int,
        depth: int,
        coef: float,
        fold_num: int,
    ) -> None:
        super().__init__()

        # init model
        self.model = AttnModel(
            input_dim=input_dim,
            out_dim=out_dim,
            attn_dim=attn_dim,
            num_heads=n_heads,
            depth=depth,
        )
        # self.model = ConvAttnModel(
        #     input_dim=input_dim,
        #     out_dim=out_dim,
        #     attn_dim=attn_dim,
        #     num_heads=n_heads,
        #     depth=depth,
        # )

        self.surv_loss = SurvLoss(alpha=0.0)
        self.coef = coef

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.fold_num = fold_num
        self.df = pd.read_pickle(
            "/home/asohn3/baraslab/germline_somatic/Data/final_surv/final_df_counts_lbls.pkl"
        )

    def final_loss(self, hazards, survival, tgt, events):
        # hazards = torch.sigmoid(t_logits)

        loss = self.surv_loss(
            hazards=hazards, survival=survival, d_lbl=tgt, event=events
        )
        loss += loss_reg_l1(self.model, self.coef)

        return loss

    def training_step(self, batch, batch_idx):
        x = batch["counts"]
        # y_p = batch['purity']
        # y_p = None
        y_lbl = batch["discrete_lbl"]
        y_e = batch["event"]

        m_out = self.model(x)

        loss = self.final_loss(
            hazards=m_out["hazards"], survival=m_out["survival"], tgt=y_lbl, events=y_e
        )

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            #  prog_bar=True,
            logger=True,
            batch_size=1,
        )

        self.training_step_outputs.append([loss, batch["tcga_id"][0]])

        return {
            "loss": loss,
        }

    def on_train_epoch_end(self):
        avg_loss = []
        tids = []
        for x in self.training_step_outputs:
            avg_loss.append(x[0])
            tids.append(x[1])
        avg_loss = torch.stack(avg_loss).mean(dim=0)

        with open(f"{self.logger.save_dir}/train_fold_{self.fold_num}.pkl", "wb") as f:
            pickle.dump(tids, f)
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
        y_lbl = batch["discrete_lbl"]
        y_e = batch["event"]

        m_out = self.model(x)

        loss = self.final_loss(
            hazards=m_out["hazards"],
            survival=m_out["survival"],
            tgt=y_lbl,
            events=y_e,
        )

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
        y_lbl = batch["discrete_lbl"]
        y_tb = batch["time_bins"]
        y_e = batch["event"]
        y_os = batch["os"]
        y_tid = batch["tcga_id"]
        cancer_type = batch["cancer_type"]

        m_out = self.model(x)

        loss = self.final_loss(
            hazards=m_out["hazards"], survival=m_out["survival"], tgt=y_lbl, events=y_e
        )

        test_out = {
            "loss": loss,
            "t_logits": m_out["t_logits"],
            "hazards": m_out["hazards"],
            "surv": m_out["survival"],
            "d_lbl": y_lbl,
            "time_bins": y_tb,
            "event": y_e,
            "os_time": y_os,
            "tcga_id": y_tid,
            "cancer_type": cancer_type,
        }
        self.test_step_outputs.append(test_out)

    @torch.no_grad()
    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()

        tids = []
        for x in self.test_step_outputs:
            tids.append(x["tcga_id"])
        with open(f"{self.logger.save_dir}/test_fold_{self.fold_num}.pkl", "wb") as f:
            pickle.dump(tids, f)

        test_t_logits = []
        test_hazards = []
        test_surv = []
        test_d_lbls = []
        test_time_bins = []
        test_os = []
        test_events = []
        test_cancer_types = []
        for x in self.test_step_outputs:
            t_logit = x["t_logits"]
            d_lbl = x["d_lbl"]
            hazards = x["hazards"]
            surv = x["surv"]
            tb = x["time_bins"]
            os = x["os_time"]
            e = x["event"]
            ct = x["cancer_type"][0]
            test_t_logits.append(t_logit)
            test_hazards.append(hazards)
            test_surv.append(surv)
            test_time_bins.append(tb)
            test_d_lbls.append(d_lbl)
            test_os.append(os)
            test_events.append(e)
            test_cancer_types.append(ct)
        test_t_logits = torch.cat(test_t_logits, dim=0)
        test_hazards = torch.cat(test_hazards, dim=0)
        test_surv = torch.cat(test_surv, dim=0)
        test_d_lbls = torch.cat(test_d_lbls, dim=0)
        test_time_bins = torch.cat(test_time_bins, dim=0)
        test_os = torch.cat(test_os, dim=0).squeeze_(-1)
        test_events = torch.cat(test_events, dim=0).squeeze_(-1)

        self.log("avg_test_loss", avg_loss, batch_size=1)

        # Survival Metrics
        test_surv_np = test_surv.cpu().numpy()
        test_os_np = test_os.cpu().numpy()
        test_events_np = test_events.cpu().numpy()

        # test_probs = nn.Softmax(dim=-1)(test_t_logits)

        N = len(test_events_np)

        risk = -1.0 * np.sum(test_surv_np, axis=1)

        # # OLD METRICS
        ci = concordance_index_censored(
            event_indicator=test_events_np.astype(np.bool_).reshape((N,)),
            event_time=test_os_np.reshape((N,)),
            estimate=risk,
        )
        # auc = self.auroc(test_probs.float(),
        #                  test_d_lbls.long().squeeze(-1))
        # auc = auc.cpu().numpy()

        test_e_n_t = create_struct_nparr(
            events=test_events_np.reshape((N,)), times=test_os_np.reshape((N,))
        )
        with open(f"{self.logger.save_dir}/train_fold_{self.fold_num}.pkl", "rb") as f:
            train_data_ids = pickle.load(f)
        train_e_n_t = get_events_time(
            df=self.df, tcga_ids=train_data_ids, which_time_metric="OS"
        )
        times_for_auc = np.percentile(
            test_os_np.reshape(
                (
                    len(
                        test_os_np,
                    )
                )
            ),
            np.linspace(21, 80, 4),
        )
        auc = cumulative_dynamic_auc(
            survival_train=train_e_n_t,
            survival_test=test_e_n_t,
            estimate=risk,
            times=times_for_auc,
        )

        print(
            f"""
            concordance_index_censored: {ci}
            cumulative dynamic AUC: {auc}
        """
        )

        self.test_results = {
            "test_loss": avg_loss.cpu().numpy(),
            "test_t_logits": test_t_logits.cpu().numpy(),
            "test_d_lbls": test_d_lbls.cpu().numpy(),
            "test_os": test_os_np,
            "test_events": test_events_np,
            "test_time_bins": test_time_bins.cpu().numpy(),
            "concordance_index": ci,
            "auroc": auc,
            "cancer_types": test_cancer_types,
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
